#!/usr/bin/env python3
"""
A* Path Planning Algorithm for TVVF-VO
Grid-based A* implementation for occupancy grid maps
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
import math
import rclpy


class Node:
    """A* ノード"""
    def __init__(self, x: int, y: int, g_cost: float = 0.0, h_cost: float = 0.0, parent=None):
        self.x = x
        self.y = y
        self.g_cost = g_cost  # スタートからこのノードまでのコスト
        self.h_cost = h_cost  # このノードからゴールまでのヒューリスティックコスト
        self.f_cost = g_cost + h_cost  # 総コスト
        self.parent = parent

    def __lt__(self, other):
        return self.f_cost < other.f_cost

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


class AStarPlanner:
    """A*経路計画器"""

    def __init__(self, safety_margin: float = 0.2):
        """
        初期化

        Args:
            safety_margin: 障害物からの安全マージン [m]
        """
        self.safety_margin = safety_margin
        self.occupancy_grid: Optional[OccupancyGrid] = None
        self.grid_data: Optional[np.ndarray] = None
        self.width = 0
        self.height = 0
        self.resolution = 0.0
        self.origin_x = 0.0
        self.origin_y = 0.0

        # 8方向の移動（隣接セル + 対角線）
        self.directions = [
            (0, 1),   # 上
            (1, 0),   # 右
            (0, -1),  # 下
            (-1, 0),  # 左
            (1, 1),   # 右上
            (1, -1),  # 右下
            (-1, 1),  # 左上
            (-1, -1)  # 左下
        ]

        # 対角線移動のコスト（√2）
        self.diagonal_cost = math.sqrt(2)

    def set_occupancy_grid(self, occupancy_grid: OccupancyGrid):
        """占有格子地図を設定"""
        self.occupancy_grid = occupancy_grid
        self.width = occupancy_grid.info.width
        self.height = occupancy_grid.info.height
        self.resolution = occupancy_grid.info.resolution
        self.origin_x = occupancy_grid.info.origin.position.x
        self.origin_y = occupancy_grid.info.origin.position.y

        # データを2D配列に変換
        self.grid_data = np.array(occupancy_grid.data).reshape((self.height, self.width))

        # 安全マージンを考慮した障害物領域の拡張
        self._expand_obstacles()

    def _expand_obstacles(self):
        """安全マージンを考慮して障害物領域を拡張"""
        if self.grid_data is None:
            return

        # 安全マージンをグリッドセル数に変換
        margin_cells = int(self.safety_margin / self.resolution)

        # 元の障害物位置を特定（未知領域も障害物として扱う）
        obstacles = (self.grid_data > 50) | (self.grid_data < 0)

        # 膨張処理
        expanded = np.copy(obstacles)
        for dx in range(-margin_cells, margin_cells + 1):
            for dy in range(-margin_cells, margin_cells + 1):
                if dx*dx + dy*dy <= margin_cells*margin_cells:
                    shifted = np.roll(np.roll(obstacles, dx, axis=1), dy, axis=0)
                    expanded |= shifted

        # 拡張された障害物マップを保存（100で占有とする）
        self.grid_data = np.where(expanded, 100, 0)

    def world_to_grid(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """ワールド座標をグリッド座標に変換"""
        grid_x = int((world_x - self.origin_x) / self.resolution)
        grid_y = int((world_y - self.origin_y) / self.resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """グリッド座標をワールド座標に変換"""
        world_x = grid_x * self.resolution + self.origin_x + self.resolution / 2
        world_y = grid_y * self.resolution + self.origin_y + self.resolution / 2
        return world_x, world_y

    def is_valid_cell(self, x: int, y: int) -> bool:
        """グリッドセルが有効かチェック"""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        if self.grid_data is None:
            return False
        # 占有率が50%以上または未知の場合は無効
        return self.grid_data[y, x] < 50

    def heuristic(self, node: Node, goal: Node) -> float:
        """ヒューリスティック関数（ユークリッド距離）"""
        dx = abs(node.x - goal.x)
        dy = abs(node.y - goal.y)
        return math.sqrt(dx*dx + dy*dy)

    def get_neighbors(self, node: Node) -> List[Node]:
        """隣接ノードを取得"""
        neighbors = []

        for i, (dx, dy) in enumerate(self.directions):
            new_x = node.x + dx
            new_y = node.y + dy

            if self.is_valid_cell(new_x, new_y):
                # 移動コスト（対角線移動は√2倍）
                move_cost = self.diagonal_cost if i >= 4 else 1.0
                g_cost = node.g_cost + move_cost

                neighbor = Node(new_x, new_y, g_cost, 0.0, node)
                neighbors.append(neighbor)

        return neighbors

    def reconstruct_path(self, goal_node: Node) -> List[Tuple[float, float]]:
        """ゴールノードから経路を再構築"""
        path = []
        current = goal_node

        while current is not None:
            world_x, world_y = self.grid_to_world(current.x, current.y)
            path.append((world_x, world_y))
            current = current.parent

        path.reverse()
        return path

    def plan_path(self, start_x: float, start_y: float,
                  goal_x: float, goal_y: float) -> Optional[List[Tuple[float, float]]]:
        """
        A*アルゴリズムで経路計画

        Args:
            start_x, start_y: スタート位置（ワールド座標）
            goal_x, goal_y: ゴール位置（ワールド座標）

        Returns:
            経路のリスト（ワールド座標）、失敗時はNone
        """
        if self.grid_data is None:
            return None

        # ワールド座標をグリッド座標に変換
        start_grid_x, start_grid_y = self.world_to_grid(start_x, start_y)
        goal_grid_x, goal_grid_y = self.world_to_grid(goal_x, goal_y)

        # スタートとゴールが有効なセルかチェック
        if not self.is_valid_cell(start_grid_x, start_grid_y):
            print(f"Start position ({start_x}, {start_y}) -> ({start_grid_x}, {start_grid_y}) is invalid")
            return None

        if not self.is_valid_cell(goal_grid_x, goal_grid_y):
            print(f"Goal position ({goal_x}, {goal_y}) -> ({goal_grid_x}, {goal_grid_y}) is invalid")
            return None

        # A*アルゴリズム
        start_node = Node(start_grid_x, start_grid_y, 0.0, 0.0)
        goal_node = Node(goal_grid_x, goal_grid_y)

        start_node.h_cost = self.heuristic(start_node, goal_node)
        start_node.f_cost = start_node.g_cost + start_node.h_cost

        open_list = [start_node]
        closed_set = set()
        open_dict = {(start_grid_x, start_grid_y): start_node}

        while open_list:
            # f_costが最小のノードを選択
            current = heapq.heappop(open_list)
            current_pos = (current.x, current.y)

            # open_dictから削除
            if current_pos in open_dict:
                del open_dict[current_pos]

            # ゴールに到達
            if current.x == goal_grid_x and current.y == goal_grid_y:
                return self.reconstruct_path(current)

            closed_set.add(current_pos)

            # 隣接ノードを探索
            for neighbor in self.get_neighbors(current):
                neighbor_pos = (neighbor.x, neighbor.y)

                if neighbor_pos in closed_set:
                    continue

                neighbor.h_cost = self.heuristic(neighbor, goal_node)
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost

                # より良い経路が見つかった場合、または新しいノードの場合
                if neighbor_pos not in open_dict or neighbor.g_cost < open_dict[neighbor_pos].g_cost:
                    open_dict[neighbor_pos] = neighbor
                    heapq.heappush(open_list, neighbor)

        # 経路が見つからない
        return None

    def smooth_path(self, path: List[Tuple[float, float]],
                   smoothing_method: str = "gaussian",
                   smoothing_iterations: int = 3,
                   smoothing_window: int = 5,
                   smoothing_factor: float = 0.8) -> List[Tuple[float, float]]:
        """
        高度な経路平滑化（複数のアルゴリズム対応）

        Args:
            path: 元の経路
            smoothing_method: 平滑化手法 ("gaussian", "weighted", "cubic_spline", "simple")
            smoothing_iterations: 平滑化の反復回数
            smoothing_window: 平滑化ウィンドウサイズ
            smoothing_factor: 平滑化強度 (0.0-1.0)

        Returns:
            平滑化された経路
        """
        if len(path) <= 2:
            return path

        if smoothing_method == "gaussian":
            return self._gaussian_smooth(path, smoothing_iterations, smoothing_window, smoothing_factor)
        elif smoothing_method == "weighted":
            return self._weighted_smooth(path, smoothing_iterations, smoothing_factor)
        elif smoothing_method == "cubic_spline":
            return self._cubic_spline_smooth(path, smoothing_factor)
        elif smoothing_method == "bezier":
            return self._bezier_smooth(path, smoothing_factor)
        else:  # "simple"
            return self._simple_smooth(path, smoothing_iterations)

    def _simple_smooth(self, path: List[Tuple[float, float]],
                      smoothing_iterations: int = 3) -> List[Tuple[float, float]]:
        """
        単純な平均化フィルタによる平滑化（元の実装）
        """
        smoothed_path = list(path)

        for _ in range(smoothing_iterations):
            new_path = [smoothed_path[0]]  # スタート点は固定

            for i in range(1, len(smoothed_path) - 1):
                # 前後の点の平均を取る
                prev_x, prev_y = smoothed_path[i-1]
                curr_x, curr_y = smoothed_path[i]
                next_x, next_y = smoothed_path[i+1]

                new_x = (prev_x + curr_x + next_x) / 3.0
                new_y = (prev_y + curr_y + next_y) / 3.0

                # 平滑化後の点が障害物内でないかチェック
                grid_x, grid_y = self.world_to_grid(new_x, new_y)
                if self.is_valid_cell(grid_x, grid_y):
                    new_path.append((new_x, new_y))
                else:
                    new_path.append((curr_x, curr_y))  # 元の点を使用

            new_path.append(smoothed_path[-1])  # ゴール点は固定
            smoothed_path = new_path

        return smoothed_path

    def _gaussian_smooth(self, path: List[Tuple[float, float]],
                        iterations: int = 3,
                        window_size: int = 5,
                        smoothing_factor: float = 0.8) -> List[Tuple[float, float]]:
        """
        ガウシアンフィルタによる平滑化
        """
        # ガウシアン重みを計算
        sigma = window_size / 6.0  # 3σ範囲をウィンドウサイズとする
        half_window = window_size // 2
        weights = []
        for i in range(-half_window, half_window + 1):
            weight = math.exp(-(i * i) / (2 * sigma * sigma))
            weights.append(weight)
        
        # 重みを正規化
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]

        smoothed_path = list(path)

        for _ in range(iterations):
            new_path = []
            
            for i in range(len(smoothed_path)):
                if i == 0 or i == len(smoothed_path) - 1:
                    # 始点と終点は固定
                    new_path.append(smoothed_path[i])
                else:
                    weighted_x = 0.0
                    weighted_y = 0.0
                    
                    for j, weight in enumerate(weights):
                        idx = i - half_window + j
                        if 0 <= idx < len(smoothed_path):
                            weighted_x += smoothed_path[idx][0] * weight
                            weighted_y += smoothed_path[idx][1] * weight
                    
                    # 元の点との混合
                    orig_x, orig_y = smoothed_path[i]
                    new_x = orig_x * (1 - smoothing_factor) + weighted_x * smoothing_factor
                    new_y = orig_y * (1 - smoothing_factor) + weighted_y * smoothing_factor
                    
                    # 障害物チェック
                    grid_x, grid_y = self.world_to_grid(new_x, new_y)
                    if self.is_valid_cell(grid_x, grid_y):
                        new_path.append((new_x, new_y))
                    else:
                        new_path.append(smoothed_path[i])
            
            smoothed_path = new_path

        return smoothed_path

    def _weighted_smooth(self, path: List[Tuple[float, float]],
                        iterations: int = 3,
                        smoothing_factor: float = 0.8) -> List[Tuple[float, float]]:
        """
        距離重み付き平滑化
        """
        smoothed_path = list(path)

        for _ in range(iterations):
            new_path = [smoothed_path[0]]  # スタート点は固定

            for i in range(1, len(smoothed_path) - 1):
                curr_x, curr_y = smoothed_path[i]
                
                # 前後の点との距離を計算
                prev_x, prev_y = smoothed_path[i-1]
                next_x, next_y = smoothed_path[i+1]
                
                prev_dist = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                next_dist = math.sqrt((curr_x - next_x)**2 + (curr_y - next_y)**2)
                
                # 距離の逆数で重み付け（近い点ほど影響大）
                if prev_dist > 0 and next_dist > 0:
                    prev_weight = 1.0 / prev_dist
                    next_weight = 1.0 / next_dist
                    total_weight = prev_weight + next_weight + 1.0
                    
                    new_x = (curr_x + prev_x * prev_weight + next_x * next_weight) / total_weight
                    new_y = (curr_y + prev_y * prev_weight + next_y * next_weight) / total_weight
                    
                    # 平滑化強度調整
                    new_x = curr_x * (1 - smoothing_factor) + new_x * smoothing_factor
                    new_y = curr_y * (1 - smoothing_factor) + new_y * smoothing_factor
                else:
                    new_x, new_y = curr_x, curr_y

                # 障害物チェック
                grid_x, grid_y = self.world_to_grid(new_x, new_y)
                if self.is_valid_cell(grid_x, grid_y):
                    new_path.append((new_x, new_y))
                else:
                    new_path.append((curr_x, curr_y))

            new_path.append(smoothed_path[-1])  # ゴール点は固定
            smoothed_path = new_path

        return smoothed_path

    def _cubic_spline_smooth(self, path: List[Tuple[float, float]],
                           smoothing_factor: float = 0.8) -> List[Tuple[float, float]]:
        """
        3次スプライン補間による平滑化
        """
        if len(path) < 4:
            return path

        # パラメータ化（累積距離）
        distances = [0.0]
        for i in range(1, len(path)):
            dist = math.sqrt((path[i][0] - path[i-1][0])**2 + (path[i][1] - path[i-1][1])**2)
            distances.append(distances[-1] + dist)

        # スプライン補間用の制御点を計算
        smoothed_path = []
        num_points = max(len(path), int(distances[-1] / (self.resolution * 2)))
        
        for i in range(num_points):
            t = (distances[-1] * i) / (num_points - 1)
            
            # t に対応する区間を見つける
            segment_idx = 0
            for j in range(len(distances) - 1):
                if distances[j] <= t <= distances[j + 1]:
                    segment_idx = j
                    break
            
            if segment_idx < len(path) - 1:
                # 3次エルミート補間
                t_norm = (t - distances[segment_idx]) / (distances[segment_idx + 1] - distances[segment_idx])
                
                p0 = path[segment_idx]
                p1 = path[segment_idx + 1]
                
                # 接線ベクトルを計算
                if segment_idx > 0:
                    t0 = ((p0[0] - path[segment_idx-1][0]) + (p1[0] - p0[0])) / 2, \
                         ((p0[1] - path[segment_idx-1][1]) + (p1[1] - p0[1])) / 2
                else:
                    t0 = (p1[0] - p0[0], p1[1] - p0[1])
                
                if segment_idx < len(path) - 2:
                    t1 = ((p1[0] - p0[0]) + (path[segment_idx+2][0] - p1[0])) / 2, \
                         ((p1[1] - p0[1]) + (path[segment_idx+2][1] - p1[1])) / 2
                else:
                    t1 = (p1[0] - p0[0], p1[1] - p0[1])
                
                # エルミート基底関数
                h00 = 2*t_norm**3 - 3*t_norm**2 + 1
                h10 = t_norm**3 - 2*t_norm**2 + t_norm
                h01 = -2*t_norm**3 + 3*t_norm**2
                h11 = t_norm**3 - t_norm**2
                
                smooth_x = h00*p0[0] + h10*t0[0] + h01*p1[0] + h11*t1[0]
                smooth_y = h00*p0[1] + h10*t0[1] + h01*p1[1] + h11*t1[1]
                
                # 元の経路との混合
                orig_weight = 1 - smoothing_factor
                if segment_idx < len(path):
                    orig_x, orig_y = path[min(segment_idx, len(path)-1)]
                    smooth_x = orig_x * orig_weight + smooth_x * smoothing_factor
                    smooth_y = orig_y * orig_weight + smooth_y * smoothing_factor
                
                # 障害物チェック
                grid_x, grid_y = self.world_to_grid(smooth_x, smooth_y)
                if self.is_valid_cell(grid_x, grid_y):
                    smoothed_path.append((smooth_x, smooth_y))
                else:
                    # 障害物がある場合は最も近い有効な元の点を使用
                    smoothed_path.append(path[min(segment_idx, len(path)-1)])

        return smoothed_path

    def _bezier_smooth(self, path: List[Tuple[float, float]],
                      smoothing_factor: float = 0.8) -> List[Tuple[float, float]]:
        """
        ベジェ曲線による平滑化
        """
        if len(path) < 4:
            return path

        smoothed_path = [path[0]]  # スタート点

        # 連続するベジェ曲線で経路を平滑化
        for i in range(0, len(path) - 1, 3):
            # 制御点を設定
            p0 = path[i]
            p3 = path[min(i + 3, len(path) - 1)]
            
            # 制御点p1, p2を中間点から計算
            if i + 1 < len(path):
                p1_base = path[i + 1]
            else:
                p1_base = ((p0[0] + p3[0]) / 2, (p0[1] + p3[1]) / 2)
            
            if i + 2 < len(path):
                p2_base = path[i + 2]
            else:
                p2_base = ((p0[0] + p3[0]) / 2, (p0[1] + p3[1]) / 2)
            
            # 制御点を調整（平滑性向上）
            p1 = (p0[0] + (p1_base[0] - p0[0]) * smoothing_factor,
                  p0[1] + (p1_base[1] - p0[1]) * smoothing_factor)
            p2 = (p3[0] + (p2_base[0] - p3[0]) * smoothing_factor,
                  p3[1] + (p2_base[1] - p3[1]) * smoothing_factor)
            
            # ベジェ曲線を生成
            num_segments = max(10, int(math.sqrt((p3[0] - p0[0])**2 + (p3[1] - p0[1])**2) / self.resolution))
            
            for j in range(1, num_segments + 1):
                t = j / num_segments
                
                # 3次ベジェ曲線
                x = (1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0]
                y = (1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1]
                
                # 障害物チェック
                grid_x, grid_y = self.world_to_grid(x, y)
                if self.is_valid_cell(grid_x, grid_y):
                    smoothed_path.append((x, y))
                else:
                    # 障害物回避：直線補間にフォールバック
                    if len(smoothed_path) > 0:
                        last_x, last_y = smoothed_path[-1]
                        smoothed_path.append(((last_x + p3[0]) / 2, (last_y + p3[1]) / 2))

        # 終点を確実に追加
        if smoothed_path[-1] != path[-1]:
            smoothed_path.append(path[-1])

        return smoothed_path

    def create_path_markers(self, path: List[Tuple[float, float]],
                           frame_id: str = "map") -> MarkerArray:
        """
        経路の可視化マーカーを作成

        Args:
            path: 経路のリスト（ワールド座標）
            frame_id: フレームID

        Returns:
            MarkerArray: 可視化マーカー
        """
        marker_array = MarkerArray()

        if not path:
            return marker_array

        # 経路ライン用マーカー
        line_marker = Marker()
        line_marker.header.frame_id = frame_id
        line_marker.header.stamp = rclpy.time.Time().to_msg()
        line_marker.ns = "astar_path"
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.05  # ライン幅
        line_marker.color.a = 1.0
        line_marker.color.r = 0.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0

        # 経路の点をラインに追加
        for x, y in path:
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.0
            line_marker.points.append(point)

        marker_array.markers.append(line_marker)

        # ウェイポイント用マーカー
        for i, (x, y) in enumerate(path):
            waypoint_marker = Marker()
            waypoint_marker.header.frame_id = frame_id
            waypoint_marker.header.stamp = rclpy.time.Time().to_msg()
            waypoint_marker.ns = "astar_waypoints"
            waypoint_marker.id = i + 1
            waypoint_marker.type = Marker.SPHERE
            waypoint_marker.action = Marker.ADD
            waypoint_marker.pose.position.x = x
            waypoint_marker.pose.position.y = y
            waypoint_marker.pose.position.z = 0.0
            waypoint_marker.pose.orientation.w = 1.0
            waypoint_marker.scale.x = 0.1
            waypoint_marker.scale.y = 0.1
            waypoint_marker.scale.z = 0.1
            waypoint_marker.color.a = 0.8

            # スタート点は青、ゴール点は赤、中間点は緑
            if i == 0:  # スタート点
                waypoint_marker.color.r = 0.0
                waypoint_marker.color.g = 0.0
                waypoint_marker.color.b = 1.0
            elif i == len(path) - 1:  # ゴール点
                waypoint_marker.color.r = 1.0
                waypoint_marker.color.g = 0.0
                waypoint_marker.color.b = 0.0
            else:  # 中間点
                waypoint_marker.color.r = 0.0
                waypoint_marker.color.g = 1.0
                waypoint_marker.color.b = 0.0

            marker_array.markers.append(waypoint_marker)

        return marker_array

    def create_path_message(self, path: List[Tuple[float, float]],
                           frame_id: str = "map") -> Path:
        """
        経路メッセージを作成

        Args:
            path: 経路のリスト（ワールド座標）
            frame_id: フレームID

        Returns:
            Path: 経路メッセージ
        """
        path_msg = Path()
        path_msg.header.frame_id = frame_id
        path_msg.header.stamp = rclpy.time.Time().to_msg()

        for i, (x, y) in enumerate(path):
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = x
            pose_stamped.pose.position.y = y
            pose_stamped.pose.position.z = 0.0

            # 向きを計算（次の点への方向）
            if i < len(path) - 1:
                next_x, next_y = path[i + 1]
                yaw = math.atan2(next_y - y, next_x - x)
            else:
                # 最後の点は前の点と同じ向き
                if i > 0:
                    prev_x, prev_y = path[i - 1]
                    yaw = math.atan2(y - prev_y, x - prev_x)
                else:
                    yaw = 0.0

            # クォータニオンに変換
            pose_stamped.pose.orientation.z = math.sin(yaw / 2.0)
            pose_stamped.pose.orientation.w = math.cos(yaw / 2.0)

            path_msg.poses.append(pose_stamped)

        return path_msg

    def create_search_visualization(self, closed_set: set, open_list: list,
                                  frame_id: str = "map") -> MarkerArray:
        """
        A*探索過程の可視化マーカーを作成

        Args:
            closed_set: 探索済みノードの集合
            open_list: 探索候補ノードのリスト
            frame_id: フレームID

        Returns:
            MarkerArray: 探索過程の可視化マーカー
        """
        marker_array = MarkerArray()

        # 探索済みノード（青色）
        if closed_set:
            closed_marker = Marker()
            closed_marker.header.frame_id = frame_id
            closed_marker.header.stamp = rclpy.time.Time().to_msg()
            closed_marker.ns = "astar_closed"
            closed_marker.id = 0
            closed_marker.type = Marker.POINTS
            closed_marker.action = Marker.ADD
            closed_marker.scale.x = 0.03
            closed_marker.scale.y = 0.03
            closed_marker.color.a = 0.5
            closed_marker.color.r = 0.0
            closed_marker.color.g = 0.0
            closed_marker.color.b = 1.0

            for x, y in closed_set:
                world_x, world_y = self.grid_to_world(x, y)
                point = Point()
                point.x = world_x
                point.y = world_y
                point.z = 0.0
                closed_marker.points.append(point)

            marker_array.markers.append(closed_marker)

        # 探索候補ノード（黄色）
        if open_list:
            open_marker = Marker()
            open_marker.header.frame_id = frame_id
            open_marker.header.stamp = rclpy.time.Time().to_msg()
            open_marker.ns = "astar_open"
            open_marker.id = 1
            open_marker.type = Marker.POINTS
            open_marker.action = Marker.ADD
            open_marker.scale.x = 0.03
            open_marker.scale.y = 0.03
            open_marker.color.a = 0.7
            open_marker.color.r = 1.0
            open_marker.color.g = 1.0
            open_marker.color.b = 0.0

            for node in open_list:
                world_x, world_y = self.grid_to_world(node.x, node.y)
                point = Point()
                point.x = world_x
                point.y = world_y
                point.z = 0.0
                open_marker.points.append(point)

            marker_array.markers.append(open_marker)

        return marker_array

    def plan_path_with_visualization(self, start_x: float, start_y: float,
                                   goal_x: float, goal_y: float,
                                   frame_id: str = "map",
                                   smoothing_method: str = "gaussian",
                                   smoothing_iterations: int = 3,
                                   smoothing_window: int = 5,
                                   smoothing_factor: float = 0.8) -> Tuple[Optional[List[Tuple[float, float]]],
                                                                  MarkerArray, Path]:
        """
        経路計画と可視化マーカーの生成（高度な平滑化対応）

        Args:
            start_x, start_y: スタート位置（ワールド座標）
            goal_x, goal_y: ゴール位置（ワールド座標）
            frame_id: フレームID
            smoothing_method: 平滑化手法 ("gaussian", "weighted", "cubic_spline", "bezier", "simple")
            smoothing_iterations: 平滑化の反復回数
            smoothing_window: ガウシアン平滑化のウィンドウサイズ
            smoothing_factor: 平滑化強度 (0.0-1.0)

        Returns:
            Tuple[経路リスト, 可視化マーカー, 経路メッセージ]
        """
        # 通常の経路計画を実行
        path = self.plan_path(start_x, start_y, goal_x, goal_y)

        # 可視化マーカーと経路メッセージを作成
        if path:
            # 高度な経路平滑化
            smoothed_path = self.smooth_path(
                path, 
                smoothing_method=smoothing_method,
                smoothing_iterations=smoothing_iterations,
                smoothing_window=smoothing_window,
                smoothing_factor=smoothing_factor
            )

            # 可視化マーカーを作成
            markers = self.create_path_markers(smoothed_path, frame_id)

            # 経路メッセージを作成
            path_msg = self.create_path_message(smoothed_path, frame_id)

            return smoothed_path, markers, path_msg
        else:
            # 経路が見つからない場合は空のマーカーとメッセージを返す
            empty_markers = MarkerArray()
            empty_path = Path()
            empty_path.header.frame_id = frame_id
            empty_path.header.stamp = rclpy.time.Time().to_msg()

            return None, empty_markers, empty_path

    def clear_markers(self, frame_id: str = "map") -> MarkerArray:
        """
        既存のマーカーをクリアするためのマーカー配列を作成

        Args:
            frame_id: フレームID

        Returns:
            MarkerArray: クリア用マーカー配列
        """
        marker_array = MarkerArray()

        # 経路ラインをクリア
        clear_line = Marker()
        clear_line.header.frame_id = frame_id
        clear_line.header.stamp = rclpy.time.Time().to_msg()
        clear_line.ns = "astar_path"
        clear_line.id = 0
        clear_line.action = Marker.DELETE
        marker_array.markers.append(clear_line)

        # ウェイポイントをクリア（最大100個と仮定）
        for i in range(100):
            clear_waypoint = Marker()
            clear_waypoint.header.frame_id = frame_id
            clear_waypoint.header.stamp = rclpy.time.Time().to_msg()
            clear_waypoint.ns = "astar_waypoints"
            clear_waypoint.id = i + 1
            clear_waypoint.action = Marker.DELETE
            marker_array.markers.append(clear_waypoint)

        return marker_array
