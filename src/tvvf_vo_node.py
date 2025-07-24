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
from nav_msgs.msg import OccupancyGrid
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
    Goal, Position, Velocity, ControlOutput, Path, AStarPathPlanner
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
        self.occupancy_grid = None  # 占有格子地図
        self.path_planner = None    # A*経路計画器
        self.planned_path = None    # 計画された経路
        self.last_planning_time = 0.0  # 最後の経路計画時刻
        self.planning_interval = 2.0   # 再計画間隔（秒）
        self.last_update_time = time.time()
        self.last_velocity = Velocity(0.0, 0.0)  # 速度計算用

        # TF2関連（高速化設定）
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # TF取得高速化用キャッシュ
        self.tf_cache = {}
        self.tf_cache_time = 0.0
        self.tf_cache_duration = 0.05  # 50msキャッシュ
        
        # 可視化高速化用タイマー
        self.last_vector_viz_time = 0.0
        self.last_path_viz_time = 0.0
        self.last_viz_time = 0.0

        # パブリッシャー
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'tvvf_vo_markers', 10)
        self.vector_field_pub = self.create_publisher(MarkerArray, 'tvvf_vo_vector_field', 10)
        self.path_pub = self.create_publisher(MarkerArray, 'planned_path', 10)

        # サブスクライバー
        # mapトピック用（transient_local QoS）
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
        map_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, map_qos
        )
        
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
        
        # A*経路統合パラメータ
        self.declare_parameter('k_path_attraction', 2.0,
                             ParameterDescriptor(description='Path following attraction strength'))
        self.declare_parameter('path_influence_radius', 2.0,
                             ParameterDescriptor(description='Path influence radius [m]'))
        self.declare_parameter('lookahead_distance', 1.5,
                             ParameterDescriptor(description='Lookahead distance [m]'))
        self.declare_parameter('path_smoothing_factor', 0.8,
                             ParameterDescriptor(description='Path smoothing factor'))
        
        # A*経路計画パラメータ
        self.declare_parameter('wall_clearance_distance', 0.8,
                             ParameterDescriptor(description='Minimum distance from walls [m]'))
        self.declare_parameter('path_inflation_radius', 0.5,
                             ParameterDescriptor(description='Path inflation radius [m]'))
        self.declare_parameter('enable_dynamic_replanning', False,
                             ParameterDescriptor(description='Enable dynamic replanning'))

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

        # ベクトル場可視化
        self.declare_parameter('enable_vector_field_viz', True,
                             ParameterDescriptor(description='Enable vector field visualization'))
        self.declare_parameter('vector_field_resolution', 0.5,
                             ParameterDescriptor(description='Vector field sampling resolution [m]'))
        self.declare_parameter('vector_field_range', 4.0,
                             ParameterDescriptor(description='Vector field visualization range [m]'))
        self.declare_parameter('vector_scale_factor', 0.3,
                             ParameterDescriptor(description='Vector arrow scale factor'))
        self.declare_parameter('max_vector_points', 500,
                             ParameterDescriptor(description='Maximum number of vector points'))
        self.declare_parameter('min_vector_magnitude', 0.05,
                             ParameterDescriptor(description='Minimum vector magnitude for display'))
        self.declare_parameter('viz_update_rate', 5.0,
                             ParameterDescriptor(description='Visualization update rate [Hz]'))
        


    def _create_config_from_parameters(self) -> TVVFVOConfig:
        """ROS2パラメータからTVVFVOConfigを作成"""
        return TVVFVOConfig(
            k_attraction=self.get_parameter('k_attraction').value,
            k_repulsion=self.get_parameter('k_repulsion').value,
            influence_radius=self.get_parameter('influence_radius').value,
            k_path_attraction=self.get_parameter('k_path_attraction').value,
            path_influence_radius=self.get_parameter('path_influence_radius').value,
            lookahead_distance=self.get_parameter('lookahead_distance').value,
            path_smoothing_factor=self.get_parameter('path_smoothing_factor').value,
            time_horizon=self.get_parameter('time_horizon').value,
            safety_margin=self.get_parameter('safety_margin').value,
            vo_resolution=self.get_parameter('vo_resolution').value,
            max_computation_time=self.get_parameter('max_computation_time').value
        )



    def _get_robot_pose_from_tf(self) -> Optional[RobotState]:
        """TFからロボット位置を取得（高速化版）"""
        try:
            current_time = time.time()
            
            # キャッシュチェック（50ms以内なら再利用）
            if (current_time - self.tf_cache_time < self.tf_cache_duration and 
                'robot_state' in self.tf_cache):
                return self.tf_cache['robot_state']
            
            base_frame = self.get_parameter('base_frame').value
            global_frame = self.get_parameter('global_frame').value

            # 高速TF取得（タイムアウト短縮）
            try:
                # ゼロタイムで即座に取得（最も高速）
                transform = self.tf_buffer.lookup_transform(
                    global_frame, base_frame, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.01)  # 10ms短縮
                )
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                # キャッシュされた値があれば使用
                if 'robot_state' in self.tf_cache:
                    self.get_logger().debug('TF取得失敗、キャッシュを使用')
                    return self.tf_cache['robot_state']
                return None

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

            # キャッシュに保存
            self.tf_cache['robot_state'] = robot_state
            self.tf_cache_time = current_time

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

            # 既存の経路をクリア（制御ループで新しい経路を計画する）
            self.planned_path = None

            self.get_logger().info(f'クリックされたポイントをゴールに設定: ({goal_position.x:.2f}, {goal_position.y:.2f}) in frame {global_frame}')
            self.get_logger().info('次の制御ループでA*経路計画を実行します')

        except Exception as e:
            self.get_logger().error(f'Clicked point callback error: {e}')

    def map_callback(self, msg: OccupancyGrid):
        """地図データコールバック"""
        try:
            self.occupancy_grid = msg
            
            # A*経路計画器を初期化
            origin = Position(
                msg.info.origin.position.x,
                msg.info.origin.position.y
            )
            wall_clearance = self.get_parameter('wall_clearance_distance').value
            self.path_planner = AStarPathPlanner(msg, msg.info.resolution, origin, wall_clearance)
            
            self.get_logger().info(f'地図データを受信: サイズ {msg.info.width}x{msg.info.height}, 解像度 {msg.info.resolution}m/cell')
            
        except Exception as e:
            self.get_logger().error(f'Map callback error: {e}')
    
    def _plan_path_to_goal(self):
        """現在位置からゴールまでの経路を計画"""
        try:
            if (self.path_planner is None or 
                self.robot_state is None or 
                self.goal is None):
                self.get_logger().warn('経路計画に必要な情報が不足しています')
                return
            
            # A*アルゴリズムで経路計画
            start_time = time.time()
            self.planned_path = self.path_planner.plan_path(
                self.robot_state.position,
                self.goal.position
            )
            planning_time = (time.time() - start_time) * 1000  # ms
            
            if self.planned_path is not None:
                path_length = len(self.planned_path.points)
                self.get_logger().info(
                    f'経路計画完了: {path_length}点, 総コスト: {self.planned_path.total_cost:.2f}m, '
                    f'計画時間: {planning_time:.1f}ms'
                )
                
                # 経路可視化の配信
                self._publish_path_visualization()
            else:
                self.get_logger().warn('経路が見つかりませんでした')
                self.planned_path = None
                
        except Exception as e:
            self.get_logger().error(f'Path planning error: {e}')
            self.planned_path = None
    
    def _check_dynamic_replanning(self):
        """動的再計画の必要性をチェック"""
        try:
            current_time = time.time()
            
            # 時間間隔チェック
            if current_time - self.last_planning_time < self.planning_interval:
                return
            
            # 経路が存在しない場合はスキップ
            if (self.planned_path is None or 
                self.robot_state is None or 
                self.goal is None or
                self.path_planner is None):
                return
            
            # 障害物が経路を阻害しているかチェック
            if self._is_path_blocked():
                self.get_logger().info('経路が阻害されました。再計画を実行します。')
                self._plan_path_to_goal()
                self.last_planning_time = current_time
            
        except Exception as e:
            self.get_logger().error(f'Dynamic replanning check error: {e}')
    
    def _is_path_blocked(self) -> bool:
        """現在の経路が障害物によって阻害されているかチェック"""
        if not self.planned_path or not self.planned_path.points:
            return False
        
        # 現在位置から少し先の経路をチェック
        robot_pos = self.robot_state.position
        check_distance = 3.0  # 3m先までチェック
        
        for path_point in self.planned_path.points:
            # 現在位置からの距離が check_distance 以内の点をチェック
            distance_from_robot = robot_pos.distance_to(path_point.position)
            if distance_from_robot > check_distance:
                continue
            
            # 各障害物が経路点に近すぎるかチェック
            for obstacle in self.obstacles:
                distance_to_obstacle = path_point.position.distance_to(obstacle.position)
                safety_distance = obstacle.radius + self.robot_state.radius + 0.5  # 安全マージン
                
                if distance_to_obstacle < safety_distance:
                    return True
        
        return False

    def laser_callback(self, msg: LaserScan):
        """レーザースキャンコールバック"""
        try:
            # 障害物検出
            self.obstacles = self._detect_obstacles_from_laser(msg)
            
            # 動的再計画チェック（設定により制御）
            if self.get_parameter('enable_dynamic_replanning').value:
                self._check_dynamic_replanning()

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

            # A*経路計画チェック（経路がない場合は計画実行）
            astar_time = 0.0
            if self.planned_path is None and self.path_planner is not None:
                self.get_logger().info('A*経路計画を実行中...')
                path_planning_start = time.time()
                self._plan_path_to_goal()
                astar_time = (time.time() - path_planning_start) * 1000  # ms
                
                if self.planned_path is not None:
                    self.get_logger().info(f'A*経路計画完了: {astar_time:.1f}ms, 経路点数: {len(self.planned_path.points)}')
                else:
                    self.get_logger().warn(f'A*経路計画失敗: {astar_time:.1f}ms - ロボットを停止します')

            # 目標到達チェック
            distance_to_goal = self.robot_state.position.distance_to(self.goal.position)
            if distance_to_goal < self.goal.tolerance:
                # 停止コマンド送信
                self._publish_stop_command()
                
                # 経路とゴールをクリア
                self.planned_path = None
                self.goal = None
                
                # 空の可視化マーカーを送信
                self._publish_empty_visualization()
                
                self.get_logger().info('ゴールに到達しました！経路とゴールをクリアしました。')
                return

            # TVVF-VO制御更新（A*経路が完了してから実行）
            tvvf_start_time = time.time()
            control_output = self.controller.update(
                self.robot_state, self.obstacles, self.goal, self.planned_path
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
            
            # デバッグ出力（常時実行）
            self._print_debug_info(stats, distance_to_goal, tf_time, tvvf_time, cmd_time)

            # 高速化された可視化（頻度制御付き）
            viz_time = self._publish_optimized_visualization()
            
            viz_time = (time.time() - viz_start_time) * 1000  # ms

            # 全体処理時間
            total_time = (time.time() - loop_start_time) * 1000  # ms
            
            # 処理時間のログ出力（A*時間も含む）
            self.get_logger().info(
                f'TVVF-VO Processing Time: '
                f'Total: {total_time:.1f}ms, '
                f'A*: {astar_time:.1f}ms, '
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
        """ベクトル場の可視化マーカーの配信（常時表示版）"""
        try:
            if self.robot_state is None:
                self.get_logger().debug('ベクトル場可視化: ロボット状態がない')
                return
            
            self.get_logger().info(f'ベクトル場可視化: ゴール={self.goal is not None}, 経路={self.planned_path is not None}')

            marker_array = MarkerArray()
            marker_id = 0

            # パラメータ取得
            resolution = self.get_parameter('vector_field_resolution').value
            viz_range = self.get_parameter('vector_field_range').value
            scale_factor = self.get_parameter('vector_scale_factor').value
            max_points = self.get_parameter('max_vector_points').value
            min_magnitude = self.get_parameter('min_vector_magnitude').value

            # ロボット位置を中心としたグリッドを作成
            robot_x = self.robot_state.position.x
            robot_y = self.robot_state.position.y

            # 高密度グリッドポイントを生成
            num_points_x = int(viz_range * 2 / resolution) + 1
            num_points_y = int(viz_range * 2 / resolution) + 1
            total_points = num_points_x * num_points_y
            
            # 適応的サンプリング：総ポイント数が制限を超える場合のみ間引き
            if total_points > max_points:
                skip_factor = max(2, int(math.sqrt(total_points / max_points) * 1.2))  # 控えめに間引き
            else:
                skip_factor = 1  # 基本解像度でサンプリング（より細かく）

            start_x = robot_x - viz_range
            start_y = robot_y - viz_range
            
            # 有効なベクトルカウンター
            valid_vector_count = 0

            # 高密度ベクトル場計算（適切な制限チェック付き）
            break_outer = False
            for i in range(0, num_points_x, skip_factor):
                for j in range(0, num_points_y, skip_factor):
                    # 制限チェック
                    if valid_vector_count >= max_points:
                        break_outer = True
                        break
                        
                    sample_x = start_x + i * resolution
                    sample_y = start_y + j * resolution

                    # ベクトル場計算
                    vector_x, vector_y = self._calculate_simple_vector_field(
                        sample_x, sample_y, robot_x, robot_y
                    )

                    # ベクトルの大きさ
                    magnitude = math.sqrt(vector_x**2 + vector_y**2)

                    # 小さすぎるベクトルはスキップ
                    if magnitude < min_magnitude:
                        continue

                    # 矢印マーカーを作成（色情報付き）
                    vector_type = self._classify_vector_type(sample_x, sample_y)
                    arrow_marker = self._create_vector_arrow_marker(
                        marker_id, sample_x, sample_y, vector_x, vector_y,
                        magnitude, scale_factor, vector_type
                    )
                    marker_array.markers.append(arrow_marker)
                    marker_id += 1
                    valid_vector_count += 1
                
                # 外側ループも適切に終了
                if break_outer:
                    break

            self.vector_field_pub.publish(marker_array)
            self.get_logger().info(f'ベクトル場可視化完了: {valid_vector_count}個のベクトルを送信')

        except Exception as e:
            self.get_logger().error(f'Vector field visualization error: {e}')

    def _calculate_simple_vector_field(self, sample_x: float, sample_y: float, 
                                     robot_x: float, robot_y: float) -> tuple:
        """簡易ベクトル場計算（A*統合版）"""
        sample_position = Position(sample_x, sample_y)
        vector_x, vector_y = 0.0, 0.0
        
        # A*経路がある場合は経路追従ベクトル場を使用
        if self.planned_path is not None and self.planned_path.points:
            path_vector = self._compute_path_vector_for_visualization(sample_position)
            vector_x += path_vector[0]
            vector_y += path_vector[1]
        
        # ゴール引力（ゴールが設定されている場合のみ）
        if self.goal is not None:
            goal_vector = self._compute_goal_vector_for_visualization(sample_position)
            vector_x += goal_vector[0]
            vector_y += goal_vector[1]
        elif self.planned_path is None:
            # ゴールも経路もない場合は、ベクトル場を表示しない
            return 0.0, 0.0

        # 障害物からの斥力
        for obstacle in self.obstacles:
            obs_x = obstacle.position.x
            obs_y = obstacle.position.y
            
            dx_obs = sample_x - obs_x
            dy_obs = sample_y - obs_y
            dist_obs = math.sqrt(dx_obs**2 + dy_obs**2)
            
            if dist_obs < obstacle.radius + 0.5:  # 影響範囲内
                if dist_obs > 0.1:
                    repulsion_strength = 0.5 / (dist_obs + 0.1)
                    vector_x += dx_obs * repulsion_strength
                    vector_y += dy_obs * repulsion_strength

        return vector_x, vector_y
    
    def _compute_path_vector_for_visualization(self, position: Position) -> tuple:
        """可視化用の経路追従ベクトル計算"""
        if not self.planned_path or not self.planned_path.points:
            return (0.0, 0.0)
        
        # 最も近い経路点を見つける
        min_distance = float('inf')
        closest_idx = 0
        
        for i, path_point in enumerate(self.planned_path.points):
            distance = position.distance_to(path_point.position)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        
        # 先読み点を計算
        lookahead_distance = self.get_parameter('lookahead_distance').value
        target_idx = closest_idx
        accumulated_distance = 0.0
        
        for i in range(closest_idx, len(self.planned_path.points) - 1):
            segment_distance = self.planned_path.points[i].position.distance_to(
                self.planned_path.points[i + 1].position
            )
            accumulated_distance += segment_distance
            
            if accumulated_distance >= lookahead_distance:
                target_idx = i + 1
                break
            else:
                target_idx = i + 1
        
        if target_idx >= len(self.planned_path.points):
            target_idx = len(self.planned_path.points) - 1
        
        # 目標点への方向ベクトル
        target_pos = self.planned_path.points[target_idx].position
        dx = target_pos.x - position.x
        dy = target_pos.y - position.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance > 0.1:
            # 経路影響範囲内でのみ有効
            path_influence = self.get_parameter('path_influence_radius').value
            if min_distance < path_influence:
                strength = self.get_parameter('k_path_attraction').value / 2.0
                influence_factor = max(0.0, (path_influence - min_distance) / path_influence)
                return (dx * strength * influence_factor, dy * strength * influence_factor)
        
        return (0.0, 0.0)
    
    def _compute_goal_vector_for_visualization(self, position: Position) -> tuple:
        """可視化用のゴール引力ベクトル計算"""
        if self.goal is None:
            return (0.0, 0.0)
        
        # A*経路がある場合は経路終端付近でのみゴール引力を適用
        if self.planned_path is not None and self.planned_path.points:
            final_path_point = self.planned_path.points[-1].position
            distance_to_final = position.distance_to(final_path_point)
            lookahead_distance = self.get_parameter('lookahead_distance').value
            
            if distance_to_final > lookahead_distance * 2:
                return (0.0, 0.0)  # 経路追従を優先
        
        # ゴールへの方向ベクトル
        dx_goal = self.goal.position.x - position.x
        dy_goal = self.goal.position.y - position.y
        dist_goal = math.sqrt(dx_goal**2 + dy_goal**2)
        
        if dist_goal > 0.1:
            attraction_strength = self.get_parameter('k_attraction').value / 2.0
            if dist_goal > self.goal.tolerance * 2:
                return (dx_goal * attraction_strength / dist_goal, 
                       dy_goal * attraction_strength / dist_goal)
            else:
                # ゴール近傍では弱い引力
                return (dx_goal * attraction_strength * 0.5, 
                       dy_goal * attraction_strength * 0.5)
        
        return (0.0, 0.0)

    def _classify_vector_type(self, sample_x: float, sample_y: float) -> str:
        """ベクトルの種類を分類（色分け用）"""
        sample_position = Position(sample_x, sample_y)
        
        # A*経路がある場合の経路追従領域チェック
        if self.planned_path is not None and self.planned_path.points:
            min_distance = float('inf')
            for path_point in self.planned_path.points:
                distance = sample_position.distance_to(path_point.position)
                min_distance = min(min_distance, distance)
            
            path_influence = self.get_parameter('path_influence_radius').value
            if min_distance < path_influence:
                return "path_following"
        
        # ゴール近傍チェック
        if self.goal is not None:
            distance_to_goal = sample_position.distance_to(self.goal.position)
            if distance_to_goal < self.goal.tolerance * 3:
                return "goal_attraction"
        
        # デフォルト
        return "general"
    
    def _create_vector_arrow_marker(self, marker_id: int, x: float, y: float,
                                  vx: float, vy: float, magnitude: float,
                                  scale_factor: float, vector_type: str = "general") -> Marker:
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

        # 矢印のスケール（細かい表示用に調整）
        marker.scale.x = 0.03  # 矢印の軸の太さ（細く）
        marker.scale.y = 0.06  # 矢印の頭の幅（小さく）
        marker.scale.z = 0.06  # 矢印の頭の高さ（小さく）

        # 色設定（ベクトルの種類に基づく）
        if vector_type == "path_following":
            # 経路追従: 緑系
            marker.color.r = 0.2
            marker.color.g = 0.8
            marker.color.b = 0.2
            marker.color.a = 0.9
        elif vector_type == "goal_attraction":
            # ゴール引力: 青系
            marker.color.r = 0.2
            marker.color.g = 0.2
            marker.color.b = 0.8
            marker.color.a = 0.9
        else:
            # 一般的なベクトル: 速度の大きさに基づく色分け
            max_magnitude = 2.0
            normalized_magnitude = min(magnitude / max_magnitude, 1.0)
            marker.color.r = normalized_magnitude
            marker.color.g = 0.0
            marker.color.b = 1.0 - normalized_magnitude
            marker.color.a = 0.8

        marker.lifetime = rclpy.duration.Duration(seconds=0.2).to_msg()

        return marker



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
        """可視化マーカーの配信（ゴールポイントのみ）"""
        try:
            # 可視化頻度制御
            current_time = time.time()
            if hasattr(self, 'last_viz_time'):
                if current_time - self.last_viz_time < 0.1:  # 0.1秒間隔
                    return
            self.last_viz_time = current_time

            marker_array = MarkerArray()
            marker_id = 0

            # 目標表示のみ
            if self.goal:
                goal_marker = self._create_goal_marker(marker_id)
                marker_array.markers.append(goal_marker)
                marker_id += 1
                self.get_logger().info(f'ゴールマーカーを作成: ({self.goal.position.x:.2f}, {self.goal.position.y:.2f})')
            else:
                self.get_logger().debug('ゴールが設定されていません')

            self.marker_pub.publish(marker_array)
            self.get_logger().debug(f'基本可視化完了: {len(marker_array.markers)}個のマーカーを送信')

        except Exception as e:
            self.get_logger().error(f'Visualization error: {e}')



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
    
    def _publish_path_visualization(self):
        """計画された経路の可視化マーカーを配信"""
        try:
            if self.planned_path is None or not self.planned_path.points:
                return
            
            marker_array = MarkerArray()
            global_frame = self.get_parameter('global_frame').value
            
            # 線分マーカー（経路全体）
            line_marker = Marker()
            line_marker.header.frame_id = global_frame
            line_marker.header.stamp = self.get_clock().now().to_msg()
            line_marker.id = 0
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            
            # 経路の点を追加
            for path_point in self.planned_path.points:
                point = Point()
                point.x = path_point.position.x
                point.y = path_point.position.y
                point.z = 0.05  # 地面から少し浮かせる
                line_marker.points.append(point)
            
            # 線の設定
            line_marker.scale.x = 0.05  # 線の太さ
            line_marker.color.r = 1.0   # 赤色
            line_marker.color.g = 0.0
            line_marker.color.b = 0.0
            line_marker.color.a = 0.8   # 透明度
            
            marker_array.markers.append(line_marker)
            
            # 経路点マーカー（サンプリング）
            marker_id = 1
            point_interval = max(1, len(self.planned_path.points) // 20)  # 最大20点まで
            
            for i in range(0, len(self.planned_path.points), point_interval):
                path_point = self.planned_path.points[i]
                
                point_marker = Marker()
                point_marker.header.frame_id = global_frame
                point_marker.header.stamp = self.get_clock().now().to_msg()
                point_marker.id = marker_id
                point_marker.type = Marker.SPHERE
                point_marker.action = Marker.ADD
                
                point_marker.pose.position.x = path_point.position.x
                point_marker.pose.position.y = path_point.position.y
                point_marker.pose.position.z = 0.05
                
                point_marker.scale.x = 0.1
                point_marker.scale.y = 0.1
                point_marker.scale.z = 0.1
                
                # 進行に従って色を変化（青→緑）
                progress = i / len(self.planned_path.points)
                point_marker.color.r = 0.0
                point_marker.color.g = progress
                point_marker.color.b = 1.0 - progress
                point_marker.color.a = 0.9
                
                marker_array.markers.append(point_marker)
                marker_id += 1
            
            self.path_pub.publish(marker_array)
            
        except Exception as e:
            self.get_logger().error(f'Path visualization error: {e}')
    
    def _publish_empty_visualization(self):
        """空の可視化マーカーを送信（経路・ゴール・ベクトル場をクリア）"""
        try:
            # 削除用マーカーを作成
            empty_marker_array = MarkerArray()
            
            # 削除マーカー（すべてのマーカーを削除）
            delete_marker = Marker()
            delete_marker.header.frame_id = "map"
            delete_marker.header.stamp = self.get_clock().now().to_msg()
            delete_marker.action = Marker.DELETEALL
            delete_marker.id = 0
            empty_marker_array.markers.append(delete_marker)
            
            # 各トピックに削除マーカーを送信
            self.marker_pub.publish(empty_marker_array)
            self.path_pub.publish(empty_marker_array)
            self.vector_field_pub.publish(empty_marker_array)
            
            self.get_logger().info('可視化マーカーをクリアしました')
            
        except Exception as e:
            self.get_logger().error(f'Empty visualization error: {e}')
    
    def _publish_optimized_visualization(self) -> float:
        """最適化された可視化処理（頻度制御付き）"""
        viz_time = 0.0
        current_time = time.time()
        viz_update_interval = 1.0 / self.get_parameter('viz_update_rate').value  # 5Hz = 0.2s間隔
        
        try:
            # 基本可視化（ゴール表示のみ）- 高頻度
            if self.get_parameter('enable_visualization').value:
                if current_time - self.last_viz_time > 0.1:  # 10Hz
                    self._publish_visualization()
                    self.last_viz_time = current_time

            # 経路可視化 - 中頻度
            if (self.planned_path is not None and 
                current_time - self.last_path_viz_time > viz_update_interval):
                self._publish_path_visualization()
                self.last_path_viz_time = current_time

            # ベクトル場可視化 - 低頻度（最も重い処理）
            if (self.get_parameter('enable_vector_field_viz').value and
                current_time - self.last_vector_viz_time > viz_update_interval):
                self._publish_vector_field_visualization()
                self.last_vector_viz_time = current_time
                
        except Exception as e:
            self.get_logger().error(f'Optimized visualization error: {e}')
        
        return viz_time




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