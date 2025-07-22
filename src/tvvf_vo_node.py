#!/usr/bin/env python3
"""
TVVF-VO ROS2ノード
Time-Varying Vector Field と Velocity Obstacles を統合したROS2ナビゲーションノード
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor

import numpy as np
from typing import List, Dict, Optional
import time
import math

# ROS2メッセージ
from geometry_msgs.msg import Twist, PoseStamped, Point, Vector3, PointStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

# TF2
import tf2_ros
import tf2_geometry_msgs
from transforms3d.euler import quat2euler, euler2quat
from geometry_msgs.msg import PointStamped as TfPointStamped

# TVVF-VOコアロジック
from .tvvf_vo_core import (
    TVVFVOController, TVVFVOConfig, RobotState, DynamicObstacle,
    Goal, Position, Velocity, ControlOutput
)


class TVVFVONode(Node):
    """TVVF-VO ROS2ナビゲーションノード"""

    def __init__(self):
        super().__init__('tvvf_vo_node')

        # パラメータ設定
        self._setup_parameters()

        # TVVF-VO制御器初期化
        self.config = self._create_config_from_parameters()
        self.controller = TVVFVOController(self.config)

        # 状態変数
        self.robot_state = None
        self.goal = None
        self.obstacles = []
        self.last_update_time = time.time()
        self.last_velocity = Velocity(0.0, 0.0)  # 速度計算用

        # TF2関連
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # パブリッシャー
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'tvvf_vo_markers', 10)
        self.debug_pub = self.create_publisher(Marker, 'tvvf_vo_debug', 10)
        self.vector_field_pub = self.create_publisher(MarkerArray, 'tvvf_vo_vector_field', 10)
        self.laser_points_pub = self.create_publisher(MarkerArray, 'tvvf_vo_laser_points', 10)

        # サブスクライバー
        # rviz2のPublish Pointツール用
        self.clicked_point_sub = self.create_subscription(
            PointStamped, 'clicked_point', self.clicked_point_callback, 10
        )
        self.laser_sub = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10
        )

        # タイマー（制御ループ）
        self.control_timer = self.create_timer(0.05, self.control_loop)  # 20Hz

        # デバッグ情報
        self.stats_history = []

        # TF確認用のタイマー
        self.tf_check_timer = self.create_timer(5.0, self._check_tf_status)

        self.get_logger().info('TVVF-VO Node initialized')

    def _setup_parameters(self):
        """ROS2パラメータの設定"""
        # TVVF関連パラメータ
        self.declare_parameter('k_attraction', 1.0,
                             ParameterDescriptor(description='Attraction field strength'))
        self.declare_parameter('k_repulsion', 2.0,
                             ParameterDescriptor(description='Repulsion field strength'))
        self.declare_parameter('influence_radius', 3.0,
                             ParameterDescriptor(description='Obstacle influence radius [m]'))

        # VO関連パラメータ
        self.declare_parameter('time_horizon', 3.0,
                             ParameterDescriptor(description='VO time horizon [s]'))
        self.declare_parameter('safety_margin', 0.2,
                             ParameterDescriptor(description='Safety margin [m]'))
        self.declare_parameter('vo_resolution', 0.1,
                             ParameterDescriptor(description='VO velocity resolution [m/s]'))

        # ロボット関連パラメータ（差動二輪用）
        self.declare_parameter('max_linear_velocity', 2.0,
                             ParameterDescriptor(description='Maximum linear velocity [m/s]'))
        self.declare_parameter('max_angular_velocity', 2.0,
                             ParameterDescriptor(description='Maximum angular velocity [rad/s]'))
        self.declare_parameter('max_acceleration', 1.0,
                             ParameterDescriptor(description='Maximum robot acceleration [m/s²]'))
        self.declare_parameter('robot_radius', 0.3,
                             ParameterDescriptor(description='Robot radius [m]'))

        # 差動二輪特有のパラメータ
        self.declare_parameter('wheel_base', 0.5,
                             ParameterDescriptor(description='Distance between wheels [m]'))
        self.declare_parameter('orientation_tolerance', 0.2,
                             ParameterDescriptor(description='Orientation tolerance for path following [rad]'))

        # フレーム名
        self.declare_parameter('base_frame', 'base_footprint',
                             ParameterDescriptor(description='Robot base frame'))
        self.declare_parameter('global_frame', 'map',
                             ParameterDescriptor(description='Global frame'))
        self.declare_parameter('laser_frame', 'lidar_link',
                             ParameterDescriptor(description='Laser scanner frame'))

        # 制御関連
        self.declare_parameter('goal_tolerance', 0.1,
                             ParameterDescriptor(description='Goal tolerance [m]'))
        self.declare_parameter('max_computation_time', 0.05,
                             ParameterDescriptor(description='Max computation time [s]'))

                # デバッグ・可視化
        self.declare_parameter('enable_visualization', True,
                             ParameterDescriptor(description='Enable visualization markers'))
        self.declare_parameter('enable_debug_output', True,  # ← Trueに変更
                             ParameterDescriptor(description='Enable debug output'))
        self.declare_parameter('enable_laser_viz', True,
                             ParameterDescriptor(description='Enable laser points visualization'))

        # ベクトル場可視化
        self.declare_parameter('enable_vector_field_viz', True,
                             ParameterDescriptor(description='Enable vector field visualization'))
        self.declare_parameter('vector_field_resolution', 0.5,
                             ParameterDescriptor(description='Vector field sampling resolution [m]'))
        self.declare_parameter('vector_field_range', 5.0,
                             ParameterDescriptor(description='Vector field visualization range [m]'))
        self.declare_parameter('vector_scale_factor', 0.5,
                             ParameterDescriptor(description='Vector arrow scale factor'))
        
        # 可視化高速化パラメータ
        self.declare_parameter('viz_update_interval', 0.1,
                             ParameterDescriptor(description='Visualization update interval [s]'))
        self.declare_parameter('vector_field_update_interval', 0.5,
                             ParameterDescriptor(description='Vector field update interval [s]'))
        self.declare_parameter('max_vector_field_points', 100,
                             ParameterDescriptor(description='Maximum vector field points'))
        self.declare_parameter('max_obstacle_markers', 10,
                             ParameterDescriptor(description='Maximum obstacle markers'))

    def _create_config_from_parameters(self) -> TVVFVOConfig:
        """ROS2パラメータからTVVFVOConfigを作成"""
        return TVVFVOConfig(
            k_attraction=self.get_parameter('k_attraction').value,
            k_repulsion=self.get_parameter('k_repulsion').value,
            influence_radius=self.get_parameter('influence_radius').value,
            time_horizon=self.get_parameter('time_horizon').value,
            safety_margin=self.get_parameter('safety_margin').value,
            vo_resolution=self.get_parameter('vo_resolution').value,
            max_computation_time=self.get_parameter('max_computation_time').value
        )



    def _get_robot_pose_from_tf(self) -> Optional[RobotState]:
        """TFからロボット位置を取得（改良版）"""
        try:
            base_frame = self.get_parameter('base_frame').value
            global_frame = self.get_parameter('global_frame').value

            # 最新の利用可能なTF変換を取得（現在時刻ではなく、最新の利用可能な時刻を使用）
            try:
                # 最初に最新のタイムスタンプを取得
                latest_time = self.tf_buffer.get_latest_common_time(global_frame, base_frame)
                transform = self.tf_buffer.lookup_transform(
                    global_frame, base_frame, latest_time,
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                # フォールバック: ゼロタイムで試行
                transform = self.tf_buffer.lookup_transform(
                    global_frame, base_frame, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )

            # 位置の取得
            position = Position(
                transform.transform.translation.x,
                transform.transform.translation.y
            )

            # 姿勢角の取得
            orientation_q = transform.transform.rotation
            _, _, yaw = quat2euler([
                orientation_q.w, orientation_q.x, orientation_q.y, orientation_q.z
            ])

            # 速度の計算（改良版数値微分）
            current_ros_time = self.get_clock().now()
            current_time = current_ros_time.nanoseconds / 1e9

            if (hasattr(self, 'last_tf_position') and hasattr(self, 'last_tf_time') and
                self.last_tf_time is not None):
                dt = current_time - self.last_tf_time
                if dt > 0.001:  # 1ms以上の差がある場合のみ計算
                    dx = position.x - self.last_tf_position.x
                    dy = position.y - self.last_tf_position.y

                    # 速度にローパスフィルタを適用（ノイズ除去）
                    new_vx = dx / dt
                    new_vy = dy / dt

                    if hasattr(self, 'last_tf_velocity'):
                        alpha = 0.3  # フィルタ係数
                        velocity = Velocity(
                            alpha * new_vx + (1 - alpha) * self.last_tf_velocity.vx,
                            alpha * new_vy + (1 - alpha) * self.last_tf_velocity.vy
                        )
                    else:
                        velocity = Velocity(new_vx, new_vy)
                else:
                    # 時間差が小さい場合は前回の速度を使用
                    velocity = getattr(self, 'last_tf_velocity', Velocity(0.0, 0.0))
            else:
                velocity = Velocity(0.0, 0.0)

            # 次回のために保存
            self.last_tf_position = position
            self.last_tf_time = current_time
            self.last_tf_velocity = velocity

            # ロボット状態作成
            robot_state = RobotState(
                position=position,
                velocity=velocity,
                orientation=yaw,
                max_velocity=self.get_parameter('max_linear_velocity').value,
                max_acceleration=self.get_parameter('max_acceleration').value,
                radius=self.get_parameter('robot_radius').value
            )

            return robot_state

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().debug(f'TF lookup failed: {e}')
            return None
        except Exception as e:
            self.get_logger().error(f'TF pose calculation error: {e}')
            return None

    def clicked_point_callback(self, msg: PointStamped):
        """クリックされたポイントをゴールとして設定（/clicked_point用）"""
        print("clicked_point_callback")
        try:
            global_frame = self.get_parameter('global_frame').value

            # フレーム変換
            if msg.header.frame_id != global_frame:
                try:
                    # TF変換を実行
                    transform_stamped = self.tf_buffer.lookup_transform(
                        global_frame, msg.header.frame_id, rclpy.time.Time()
                    )

                    # ポイントをグローバルフレームに変換
                    point_in_global = tf2_geometry_msgs.do_transform_point(
                        msg, transform_stamped
                    )
                    goal_x = point_in_global.point.x
                    goal_y = point_in_global.point.y

                except Exception as tf_e:
                    self.get_logger().warn(f'TF transform failed for clicked point: {tf_e}, using original coordinates')
                    goal_x = msg.point.x
                    goal_y = msg.point.y
            else:
                goal_x = msg.point.x
                goal_y = msg.point.y

            goal_position = Position(goal_x, goal_y)

            self.goal = Goal(
                position=goal_position,
                tolerance=self.get_parameter('goal_tolerance').value
            )

            self.get_logger().info(f'クリックされたポイントをゴールに設定: ({goal_position.x:.2f}, {goal_position.y:.2f}) in frame {global_frame}')

        except Exception as e:
            self.get_logger().error(f'Clicked point callback error: {e}')

    def laser_callback(self, msg: LaserScan):
        """レーザースキャンコールバック"""
        try:
            # 障害物検出
            self.obstacles = self._detect_obstacles_from_laser(msg)

            # レーザーポイントの可視化
            if self.get_parameter('enable_laser_viz').value:
                self._publish_laser_points_visualization(msg)

        except Exception as e:
            self.get_logger().error(f'Laser callback error: {e}')

    def _detect_obstacles_from_laser(self, laser_msg: LaserScan) -> List[DynamicObstacle]:
        """レーザースキャンから障害物検出（修正版）"""
        obstacles = []

        # クラスタリングによる障害物検出
        min_distance = 0.5  # 最小検出距離
        max_distance = self.config.influence_radius
        cluster_threshold = 0.3  # クラスタリング閾値

        valid_points = []

        # レーザーフレームとグローバルフレームの取得
        laser_frame = laser_msg.header.frame_id
        global_frame = self.get_parameter('global_frame').value

        # デバッグ情報を追加
        self.get_logger().debug(f'Laser frame: {laser_frame}, Global frame: {global_frame}')

        # TF変換の準備（レーザーメッセージのタイムスタンプを使用）
        try:
            # レーザーメッセージのタイムスタンプを使用
            laser_time = rclpy.time.Time.from_msg(laser_msg.header.stamp)
            
            # TF変換の可用性をチェック
            if not self.tf_buffer.can_transform(global_frame, laser_frame, laser_time):
                self.get_logger().warn(f'TF transform not available: {global_frame} -> {laser_frame}')
                return obstacles
                
            transform_stamped = self.tf_buffer.lookup_transform(
                global_frame, laser_frame, laser_time,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            self.get_logger().debug(f'TF transform successful: {len(laser_msg.ranges)} points')
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'TF lookup failed for laser: {e}')
            return obstacles

        # 有効なポイント数をカウント
        valid_count = 0
        
        for i, distance in enumerate(laser_msg.ranges):
            if min_distance < distance < max_distance and not math.isinf(distance) and not math.isnan(distance):
                angle = laser_msg.angle_min + i * laser_msg.angle_increment

                # レーザーフレームでの障害物位置
                x_laser = distance * math.cos(angle)
                y_laser = distance * math.sin(angle)

                # TF2を使ってグローバルフレームに変換
                try:
                    # PointStampedメッセージを作成
                    laser_point = TfPointStamped()
                    laser_point.header.frame_id = laser_frame
                    laser_point.header.stamp = laser_msg.header.stamp
                    laser_point.point.x = x_laser
                    laser_point.point.y = y_laser
                    laser_point.point.z = 0.0

                    # グローバルフレームに変換
                    global_point = tf2_geometry_msgs.do_transform_point(
                        laser_point, transform_stamped
                    )

                    valid_points.append((global_point.point.x, global_point.point.y))
                    valid_count += 1

                except Exception as e:
                    self.get_logger().debug(f'Point transform failed: {e}')
                    continue

        self.get_logger().debug(f'Valid points: {valid_count}/{len(laser_msg.ranges)}')

        # 簡易クラスタリング
        clusters = self._cluster_points(valid_points, cluster_threshold)

        # 各クラスタを障害物として追加
        for i, cluster in enumerate(clusters):
            if len(cluster) >= 3:  # 最小ポイント数
                # クラスタの重心を計算
                center_x = sum(p[0] for p in cluster) / len(cluster)
                center_y = sum(p[1] for p in cluster) / len(cluster)

                # 障害物半径の推定（簡易）
                distances = [math.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2)
                           for p in cluster]
                radius = max(distances) + 0.1  # 安全マージン

                obstacles.append(DynamicObstacle(
                    id=i,
                    position=Position(center_x, center_y),
                    velocity=Velocity(0.0, 0.0),  # 静的障害物として扱う
                    radius=min(radius, 0.5)  # 最大半径制限
                ))

        self.get_logger().debug(f'Detected obstacles: {len(obstacles)}')
        return obstacles

    def _cluster_points(self, points: List[tuple], threshold: float) -> List[List[tuple]]:
        """簡易クラスタリング"""
        if not points:
            return []

        clusters = []
        visited = [False] * len(points)

        for i, point in enumerate(points):
            if visited[i]:
                continue

            cluster = [point]
            visited[i] = True

            # 近隣点を探索
            for j, other_point in enumerate(points):
                if not visited[j]:
                    distance = math.sqrt(
                        (point[0] - other_point[0])**2 +
                        (point[1] - other_point[1])**2
                    )
                    if distance < threshold:
                        cluster.append(other_point)
                        visited[j] = True

            clusters.append(cluster)

        return clusters

    def control_loop(self):
        """メイン制御ループ"""
        loop_start_time = time.time()  # 全体処理時間計測開始
        
        try:
            # ロボット状態の更新（TFから取得）
            tf_start_time = time.time()
            tf_robot_state = self._get_robot_pose_from_tf()
            tf_time = (time.time() - tf_start_time) * 1000  # ms
            
            if tf_robot_state is not None:
                self.robot_state = tf_robot_state

            # 状態チェック
            if self.robot_state is None or self.goal is None:
                return

            # 目標到達チェック
            distance_to_goal = self.robot_state.position.distance_to(self.goal.position)
            if distance_to_goal < self.goal.tolerance:
                # 停止コマンド送信
                self._publish_stop_command()
                self.get_logger().info('ゴールに到達しました！')
                return

            # TVVF-VO制御更新
            tvvf_start_time = time.time()
            control_output = self.controller.update(
                self.robot_state, self.obstacles, self.goal
            )
            tvvf_time = (time.time() - tvvf_start_time) * 1000  # ms

            # 制御コマンド発行
            cmd_start_time = time.time()
            self._publish_control_command(control_output)
            cmd_time = (time.time() - cmd_start_time) * 1000  # ms

            # 統計情報更新
            stats = self.controller.get_stats()
            self.stats_history.append(stats)

            # 可視化処理時間計測
            viz_start_time = time.time()
            
            # デバッグ出力
            if self.get_parameter('enable_debug_output').value:
                self._print_debug_info(stats, distance_to_goal, tf_time, tvvf_time, cmd_time)

            # 可視化
            if self.get_parameter('enable_visualization').value:
                self._publish_visualization()

            # ベクトル場可視化
            if self.get_parameter('enable_vector_field_viz').value:
                self._publish_vector_field_visualization()
            
            viz_time = (time.time() - viz_start_time) * 1000  # ms

            # 全体処理時間
            total_time = (time.time() - loop_start_time) * 1000  # ms
            
            # 処理時間のログ出力（常時出力）
            self.get_logger().info(
                f'TVVF-VO Processing Time: '
                f'Total: {total_time:.1f}ms, '
                f'TF: {tf_time:.1f}ms, '
                f'TVVF: {tvvf_time:.1f}ms, '
                f'Cmd: {cmd_time:.1f}ms, '
                f'Viz: {viz_time:.1f}ms'
            )

        except Exception as e:
            self.get_logger().error(f'Control loop error: {e}')
            self._publish_stop_command()

    def _publish_control_command(self, control_output: ControlOutput):
        """制御コマンドの配信（差動二輪用）"""
        # ベクトル場からの速度指令を差動二輪の制約に変換
        desired_vx = control_output.velocity_command.vx
        desired_vy = control_output.velocity_command.vy

        # 差動二輪用の速度指令に変換
        linear_x, angular_z = self._convert_to_differential_drive(
            desired_vx, desired_vy, self.robot_state.orientation
        )

        cmd_msg = Twist()
        cmd_msg.linear.x = linear_x
        cmd_msg.linear.y = 0.0  # 差動二輪では横移動不可
        cmd_msg.linear.z = 0.0
        cmd_msg.angular.x = 0.0
        cmd_msg.angular.y = 0.0
        cmd_msg.angular.z = angular_z

        # 速度制限
        max_linear_vel = self.get_parameter('max_linear_velocity').value
        max_angular_vel = self.get_parameter('max_angular_velocity').value

        # 線形速度制限
        if abs(cmd_msg.linear.x) > max_linear_vel:
            cmd_msg.linear.x = max_linear_vel if cmd_msg.linear.x > 0 else -max_linear_vel

        # 角速度制限
        if abs(cmd_msg.angular.z) > max_angular_vel:
            cmd_msg.angular.z = max_angular_vel if cmd_msg.angular.z > 0 else -max_angular_vel

        self.cmd_vel_pub.publish(cmd_msg)

    def _convert_to_differential_drive(self, desired_vx: float, desired_vy: float,
                                     current_orientation: float) -> tuple:
        """ベクトル場からの速度指令を差動二輪用に変換"""
        # 目標速度ベクトルの大きさと方向
        target_speed = math.sqrt(desired_vx**2 + desired_vy**2)
        target_angle = math.atan2(desired_vy, desired_vx)

        # 現在の姿勢との角度差
        angle_diff = target_angle - current_orientation

        # 角度を-piからpiの範囲に正規化
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        orientation_tolerance = self.get_parameter('orientation_tolerance').value

        # 角度差が大きい場合は回転を優先
        if abs(angle_diff) > orientation_tolerance:
            # 回転優先モード
            linear_velocity = target_speed * math.cos(angle_diff) * 0.3  # 減速
            angular_velocity = 2.0 * angle_diff  # 角度差に比例した角速度
        else:
            # 前進優先モード
            linear_velocity = target_speed * math.cos(angle_diff)
            angular_velocity = 1.0 * angle_diff  # 小さな補正

        return linear_velocity, angular_velocity

    def _publish_stop_command(self):
        """停止コマンドの配信"""
        cmd_msg = Twist()
        self.cmd_vel_pub.publish(cmd_msg)

    def _print_debug_info(self, stats: Dict, distance_to_goal: float, 
                         tf_time: float, tvvf_time: float, cmd_time: float):
        """デバッグ情報の出力（処理時間詳細版）"""
        self.get_logger().info(
            f'TVVF-VO Debug: '
            f'Core computation: {stats["computation_time"]:.1f}ms, '
            f'TF lookup: {tf_time:.1f}ms, '
            f'TVVF update: {tvvf_time:.1f}ms, '
            f'Command publish: {cmd_time:.1f}ms, '
            f'VO cones: {stats["num_vo_cones"]}, '
            f'Safety margin: {stats["safety_margin"]:.2f}m, '
            f'Goal distance: {distance_to_goal:.2f}m'
        )

    def _publish_vector_field_visualization(self):
        """ベクトル場の可視化マーカーの配信（高速化版）"""
        try:
            if self.robot_state is None or self.goal is None:
                return

            marker_array = MarkerArray()
            marker_id = 0

            # パラメータ取得
            resolution = self.get_parameter('vector_field_resolution').value
            viz_range = self.get_parameter('vector_field_range').value
            scale_factor = self.get_parameter('vector_scale_factor').value

            # 可視化頻度制御（毎回ではなく間隔を空ける）
            current_time = time.time()
            if hasattr(self, 'last_vector_field_time'):
                if current_time - self.last_vector_field_time < 0.5:  # 0.5秒間隔
                    return
            self.last_vector_field_time = current_time

            # ロボット位置を中心としたグリッドを作成
            robot_x = self.robot_state.position.x
            robot_y = self.robot_state.position.y

            # グリッドポイントを生成（解像度を上げて計算量を削減）
            num_points = int(viz_range * 2 / resolution) + 1
            
            # 計算量削減のため、グリッドポイント数を制限
            max_points = 100  # 最大100ポイントに制限
            if num_points * num_points > max_points:
                # 解像度を動的に調整
                resolution = math.sqrt((viz_range * 2) ** 2 / max_points)
                num_points = int(viz_range * 2 / resolution) + 1

            start_x = robot_x - viz_range
            start_y = robot_y - viz_range

            # ベクトル場計算の高速化
            for i in range(0, num_points, 2):  # 2つおきにサンプリング
                for j in range(0, num_points, 2):
                    sample_x = start_x + i * resolution
                    sample_y = start_y + j * resolution

                    # 簡易ベクトル場計算（TVVF-VO制御器を使わない高速版）
                    vector_x, vector_y = self._calculate_simple_vector_field(
                        sample_x, sample_y, robot_x, robot_y
                    )

                    # ベクトルの大きさ
                    magnitude = math.sqrt(vector_x**2 + vector_y**2)

                    # 小さすぎるベクトルはスキップ
                    if magnitude < 0.01:
                        continue

                    # 矢印マーカーを作成
                    arrow_marker = self._create_vector_arrow_marker(
                        marker_id, sample_x, sample_y, vector_x, vector_y,
                        magnitude, scale_factor
                    )
                    marker_array.markers.append(arrow_marker)
                    marker_id += 1

            self.vector_field_pub.publish(marker_array)

        except Exception as e:
            self.get_logger().error(f'Vector field visualization error: {e}')

    def _calculate_simple_vector_field(self, sample_x: float, sample_y: float, 
                                     robot_x: float, robot_y: float) -> tuple:
        """簡易ベクトル場計算（高速版）"""
        # 目標への引力
        if self.goal is not None:
            goal_x = self.goal.position.x
            goal_y = self.goal.position.y
            
            # 目標への方向ベクトル
            dx_goal = goal_x - sample_x
            dy_goal = goal_y - sample_y
            dist_goal = math.sqrt(dx_goal**2 + dy_goal**2)
            
            if dist_goal > 0.1:
                # 引力（距離に反比例）
                attraction_strength = 1.0 / (1.0 + dist_goal)
                vector_x = dx_goal * attraction_strength
                vector_y = dy_goal * attraction_strength
            else:
                vector_x = 0.0
                vector_y = 0.0
        else:
            vector_x = 0.0
            vector_y = 0.0

        # 障害物からの斥力（簡易版）
        for obstacle in self.obstacles:
            obs_x = obstacle.position.x
            obs_y = obstacle.position.y
            
            # 障害物への方向ベクトル
            dx_obs = sample_x - obs_x
            dy_obs = sample_y - obs_y
            dist_obs = math.sqrt(dx_obs**2 + dy_obs**2)
            
            if dist_obs < obstacle.radius + 0.5:  # 影響範囲内
                if dist_obs > 0.1:
                    # 斥力（距離の逆数）
                    repulsion_strength = 0.5 / (dist_obs + 0.1)
                    vector_x += dx_obs * repulsion_strength
                    vector_y += dy_obs * repulsion_strength

        return vector_x, vector_y

    def _create_vector_arrow_marker(self, marker_id: int, x: float, y: float,
                                  vx: float, vy: float, magnitude: float,
                                  scale_factor: float) -> Marker:
        """ベクトル矢印マーカー作成"""
        marker = Marker()
        marker.header.frame_id = self.get_parameter('global_frame').value
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # 矢印の始点と終点を設定
        start_point = Point()
        start_point.x = x
        start_point.y = y
        start_point.z = 0.0

        end_point = Point()
        end_point.x = x + vx * scale_factor
        end_point.y = y + vy * scale_factor
        end_point.z = 0.0

        marker.points = [start_point, end_point]

        # 矢印のスケール
        marker.scale.x = 0.05  # 矢印の軸の太さ
        marker.scale.y = 0.1   # 矢印の頭の幅
        marker.scale.z = 0.1   # 矢印の頭の高さ

        # 色設定（速度の大きさに基づく）
        # 青（低速）から赤（高速）へのグラデーション
        max_magnitude = 2.0  # 想定最大速度
        normalized_magnitude = min(magnitude / max_magnitude, 1.0)

        marker.color.r = normalized_magnitude
        marker.color.g = 0.0
        marker.color.b = 1.0 - normalized_magnitude
        marker.color.a = 0.8

        marker.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()

        return marker

    def _publish_laser_points_visualization(self, laser_msg: LaserScan):
        """レーザーポイントの可視化（修正版）"""
        try:
            marker_array = MarkerArray()

            # レーザーフレームとグローバルフレームの取得
            laser_frame = laser_msg.header.frame_id
            global_frame = self.get_parameter('global_frame').value

            # TF変換の準備（レーザーメッセージのタイムスタンプを使用）
            try:
                laser_time = rclpy.time.Time.from_msg(laser_msg.header.stamp)
                
                # TF変換の可用性をチェック
                if not self.tf_buffer.can_transform(global_frame, laser_frame, laser_time):
                    self.get_logger().warn(f'TF transform not available for visualization: {global_frame} -> {laser_frame}')
                    return
                    
                transform_stamped = self.tf_buffer.lookup_transform(
                    global_frame, laser_frame, laser_time,
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                self.get_logger().debug(f'TF lookup failed for laser visualization: {e}')
                return

            # レーザーポイントのマーカー作成
            points_marker = Marker()
            points_marker.header.frame_id = global_frame
            points_marker.header.stamp = laser_msg.header.stamp
            points_marker.id = 0
            points_marker.type = Marker.POINTS
            points_marker.action = Marker.ADD
            points_marker.scale.x = 0.05  # ポイントサイズ
            points_marker.scale.y = 0.05
            points_marker.color.r = 1.0
            points_marker.color.g = 1.0
            points_marker.color.b = 0.0  # 黄色
            points_marker.color.a = 0.8
            points_marker.lifetime = rclpy.duration.Duration(seconds=0.1).to_msg()

            # レーザーポイントを変換してマーカーに追加
            point_count = 0
            for i, distance in enumerate(laser_msg.ranges):
                if 0.1 < distance < 10.0 and not math.isinf(distance) and not math.isnan(distance):
                    angle = laser_msg.angle_min + i * laser_msg.angle_increment

                    # レーザーフレームでの位置
                    x_laser = distance * math.cos(angle)
                    y_laser = distance * math.sin(angle)

                    try:
                        # TF2を使ってグローバルフレームに変換
                        laser_point = TfPointStamped()
                        laser_point.header.frame_id = laser_frame
                        laser_point.header.stamp = laser_msg.header.stamp
                        laser_point.point.x = x_laser
                        laser_point.point.y = y_laser
                        laser_point.point.z = 0.0

                        global_point = tf2_geometry_msgs.do_transform_point(
                            laser_point, transform_stamped
                        )

                        # マーカーポイントに追加
                        point = Point()
                        point.x = global_point.point.x
                        point.y = global_point.point.y
                        point.z = 0.0
                        points_marker.points.append(point)
                        point_count += 1

                    except Exception as e:
                        continue

            if point_count > 0:
                marker_array.markers.append(points_marker)
                self.laser_points_pub.publish(marker_array)
                self.get_logger().debug(f'Published {point_count} laser points')

        except Exception as e:
            self.get_logger().error(f'Laser visualization error: {e}')

    def _check_tf_status(self):
        """TF状態を定期的にチェック（デバッグ用）"""
        try:
            base_frame = self.get_parameter('base_frame').value
            global_frame = self.get_parameter('global_frame').value
            laser_frame = self.get_parameter('laser_frame').value

            # フレーム間の変換が利用可能かチェック
            tf_status = []

            # 1. グローバル → ベース
            if self.tf_buffer.can_transform(global_frame, base_frame, rclpy.time.Time()):
                tf_status.append(f"{global_frame} -> {base_frame}: OK")
            else:
                tf_status.append(f"{global_frame} -> {base_frame}: NG")

            # 2. グローバル → レーザー
            if self.tf_buffer.can_transform(global_frame, laser_frame, rclpy.time.Time()):
                tf_status.append(f"{global_frame} -> {laser_frame}: OK")
            else:
                tf_status.append(f"{global_frame} -> {laser_frame}: NG")

            # データソース確認
            if self.robot_state is not None:
                self.get_logger().info(f'ロボット位置: TF使用, TF状態: {", ".join(tf_status)}')
            else:
                self.get_logger().warn(f'ロボット位置: データなし, TF状態: {", ".join(tf_status)}')

        except Exception as e:
            self.get_logger().debug(f'TF status check error: {e}')

    def _publish_visualization(self):
        """可視化マーカーの配信（高速化版）"""
        try:
            # 可視化頻度制御
            current_time = time.time()
            if hasattr(self, 'last_viz_time'):
                if current_time - self.last_viz_time < 0.1:  # 0.1秒間隔
                    return
            self.last_viz_time = current_time

            marker_array = MarkerArray()
            marker_id = 0

            # ロボット表示
            if self.robot_state:
                robot_marker = self._create_robot_marker(marker_id)
                marker_array.markers.append(robot_marker)
                marker_id += 1

            # 目標表示
            if self.goal:
                goal_marker = self._create_goal_marker(marker_id)
                marker_array.markers.append(goal_marker)
                marker_id += 1

            # 障害物表示（最大10個まで制限）
            obstacle_count = 0
            for obstacle in self.obstacles:
                if obstacle_count >= 10:  # 最大10個の障害物のみ表示
                    break
                obstacle_marker = self._create_obstacle_marker(obstacle, marker_id)
                marker_array.markers.append(obstacle_marker)
                marker_id += 1
                obstacle_count += 1

            self.marker_pub.publish(marker_array)

        except Exception as e:
            self.get_logger().error(f'Visualization error: {e}')

    def _create_robot_marker(self, marker_id: int) -> Marker:
        """ロボットマーカー作成"""
        marker = Marker()
        marker.header.frame_id = self.get_parameter('global_frame').value
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = self.robot_state.position.x
        marker.pose.position.y = self.robot_state.position.y
        marker.pose.position.z = 0.0

        quat = euler2quat(0, 0, self.robot_state.orientation)
        marker.pose.orientation.w = quat[0]
        marker.pose.orientation.x = quat[1]
        marker.pose.orientation.y = quat[2]
        marker.pose.orientation.z = quat[3]

        marker.scale.x = self.robot_state.radius * 2
        marker.scale.y = self.robot_state.radius * 2
        marker.scale.z = 0.1

        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.8

        return marker

    def _create_goal_marker(self, marker_id: int) -> Marker:
        """目標マーカー作成"""
        marker = Marker()
        marker.header.frame_id = self.get_parameter('global_frame').value
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = self.goal.position.x
        marker.pose.position.y = self.goal.position.y
        marker.pose.position.z = 0.0

        marker.scale.x = self.goal.tolerance * 2
        marker.scale.y = self.goal.tolerance * 2
        marker.scale.z = 0.05

        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8

        return marker

    def _create_obstacle_marker(self, obstacle: DynamicObstacle, marker_id: int) -> Marker:
        """障害物マーカー作成"""
        marker = Marker()
        marker.header.frame_id = self.get_parameter('global_frame').value
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = obstacle.position.x
        marker.pose.position.y = obstacle.position.y
        marker.pose.position.z = 0.0

        marker.scale.x = obstacle.radius * 2
        marker.scale.y = obstacle.radius * 2
        marker.scale.z = 0.2

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.7

        return marker


def main(args=None):
    """メイン関数"""
    rclpy.init(args=args)

    try:
        node = TVVFVONode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Node error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()