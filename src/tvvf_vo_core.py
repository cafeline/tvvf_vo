#!/usr/bin/env python3
"""
TVVF + Velocity Obstacles統合システム - コアロジック
Time-Varying Vector Field と Velocity Obstacles を統合した動的環境ナビゲーション
ROS2から独立したコアロジック
"""

import numpy as np
import copy
import time
import heapq
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union
from enum import Enum

# ============================================================================
# 基本データクラス定義
# ============================================================================

@dataclass
class Position:
    """2D位置クラス"""
    x: float
    y: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def distance_to(self, other: 'Position') -> float:
        return np.linalg.norm(self.to_array() - other.to_array())

    def __add__(self, other: 'Position') -> 'Position':
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Position') -> 'Position':
        return Position(self.x - other.x, self.y - other.y)

@dataclass
class Velocity:
    """2D速度クラス"""
    vx: float
    vy: float

    def to_array(self) -> np.ndarray:
        return np.array([self.vx, self.vy])

    def magnitude(self) -> float:
        return np.linalg.norm(self.to_array())

    def normalize(self) -> 'Velocity':
        mag = self.magnitude()
        if mag < 1e-8:
            return Velocity(0.0, 0.0)
        return Velocity(self.vx / mag, self.vy / mag)

@dataclass
class RobotState:
    """ロボット状態クラス"""
    position: Position
    velocity: Velocity
    orientation: float  # [rad]
    max_velocity: float = 2.0  # [m/s]
    max_acceleration: float = 1.0  # [m/s²]
    radius: float = 0.3  # [m]

@dataclass
class DynamicObstacle:
    """動的障害物クラス"""
    id: int
    position: Position
    velocity: Velocity
    radius: float
    prediction_horizon: float = 3.0  # [s]
    uncertainty: float = 0.1  # [m]

    def predict_position(self, time_delta: float) -> Position:
        """等速度モデルによる位置予測"""
        predicted_x = self.position.x + self.velocity.vx * time_delta
        predicted_y = self.position.y + self.velocity.vy * time_delta
        return Position(predicted_x, predicted_y)

@dataclass
class Goal:
    """目標クラス"""
    position: Position
    tolerance: float = 0.1  # [m]

@dataclass
class PathPoint:
    """経路上の点クラス"""
    position: Position
    cost: float = 0.0
    
@dataclass
class Path:
    """経路クラス"""
    points: List[PathPoint] = field(default_factory=list)
    total_cost: float = 0.0
    
    def add_point(self, position: Position, cost: float = 0.0):
        """経路に点を追加"""
        self.points.append(PathPoint(position, cost))
        self.total_cost += cost

@dataclass
class ControlOutput:
    """制御出力クラス"""
    velocity_command: Velocity
    angular_velocity: float = 0.0  # [rad/s]
    execution_time: float = 0.05  # [s]
    safety_margin: float = 0.0  # [m]

# ============================================================================
# 設定クラス
# ============================================================================

@dataclass
class TVVFVOConfig:
    """TVVF-VO統合システム設定"""
    # TVVF関連
    k_attraction: float = 1.0
    k_repulsion: float = 2.0
    influence_radius: float = 3.0

    # A*経路統合関連
    k_path_attraction: float = 2.0     # 経路追従の引力強度
    path_influence_radius: float = 2.0  # 経路の影響半径
    lookahead_distance: float = 1.5    # 先読み距離
    path_smoothing_factor: float = 0.8  # 経路スムージング係数

    # VO関連
    time_horizon: float = 3.0
    safety_margin: float = 0.2
    vo_resolution: float = 0.1

    # 予測関連
    prediction_dt: float = 0.1
    uncertainty_growth: float = 0.1

    # 最適化関連
    direction_weight: float = 1.0
    safety_weight: float = 2.0
    efficiency_weight: float = 0.5

    # 数値安定性
    min_distance: float = 1e-6
    max_force: float = 10.0

    # 性能関連
    max_computation_time: float = 0.05  # [s]

# ============================================================================
# ユーティリティ関数
# ============================================================================

def safe_normalize(vector: np.ndarray, min_norm: float = 1e-8) -> np.ndarray:
    """安全な正規化（ゼロ除算回避）"""
    norm = np.linalg.norm(vector)
    if norm < min_norm:
        return np.zeros_like(vector)
    return vector / norm

def safe_normalize_with_default(vector: np.ndarray, min_norm: float = 1e-8, default_value: np.ndarray = None) -> np.ndarray:
    """安全な正規化（デフォルト値指定可能）"""
    norm = np.linalg.norm(vector)
    if norm < min_norm:
        if default_value is not None:
            return default_value
        return np.zeros_like(vector)
    return vector / norm

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全な除算（ゼロ除算回避）"""
    if abs(denominator) < 1e-8:
        return default
    return numerator / denominator

def clip_magnitude(vector: np.ndarray, max_magnitude: float) -> np.ndarray:
    """ベクトルの大きさを制限"""
    magnitude = np.linalg.norm(vector)
    if magnitude > max_magnitude:
        return vector * (max_magnitude / magnitude)
    return vector

# ============================================================================
# TVVF計算クラス
# ============================================================================

class TVVFGenerator:
    """Time-Varying Vector Field生成器"""

    def __init__(self, config: TVVFVOConfig):
        self.config = config

    def compute_vector(self, position: Position, time: float,
                      goal: Goal, obstacles: List[DynamicObstacle],
                      planned_path: Optional[Path] = None) -> np.ndarray:
        """TVVF計算関数（A*経路統合版）"""
        
        # 1. A*経路の存在チェック（必須）
        if planned_path is None or not planned_path.points:
            # A*経路がない場合はゼロベクトルを返す
            return np.zeros(2)
            
        # A*経路追従ベクトル場を使用（必須）
        path_force = self._compute_path_following_force(position, planned_path)
        goal_force = self._compute_adaptive_goal_force(position, goal, planned_path)

        # 2. 斥力場計算（時間依存・従来通り）
        repulsive_force = self._compute_repulsive_force(position, time, obstacles)

        # 3. 時間依存補正項
        time_correction = self._compute_time_correction(position, time, obstacles, goal)

        # 4. 合成（A*経路を優先）
        total_force = path_force + goal_force + repulsive_force + time_correction

        # 5. 数値安定性とクリッピング
        total_force = clip_magnitude(total_force, self.config.max_force)

        return total_force

    def _compute_attractive_force(self, position: Position, goal: Goal) -> np.ndarray:
        """引力場計算"""
        goal_vector = goal.position.to_array() - position.to_array()
        distance = np.linalg.norm(goal_vector)

        if distance < self.config.min_distance:
            return np.zeros(2)

        if distance > goal.tolerance * 2:
            attractive_force = self.config.k_attraction * safe_normalize(goal_vector)
        else:
            attractive_force = self.config.k_attraction * goal_vector * 0.5

        return attractive_force

    def _compute_repulsive_force(self, position: Position, time: float,
                                obstacles: List[DynamicObstacle]) -> np.ndarray:
        """時間依存斥力場計算"""
        total_repulsive = np.zeros(2)

        for obstacle in obstacles:
            potential_gradient = self._compute_dynamic_potential_gradient(
                position, obstacle, time
            )
            total_repulsive += potential_gradient

        return total_repulsive

    def _compute_dynamic_potential_gradient(self, position: Position,
                                          obstacle: DynamicObstacle,
                                          time: float) -> np.ndarray:
        """個別障害物の動的ポテンシャル勾配計算"""
        current_relative_pos = position.to_array() - obstacle.position.to_array()
        current_distance = np.linalg.norm(current_relative_pos)

        if current_distance > self.config.influence_radius:
            return np.zeros(2)

        min_distance_time = self._compute_minimum_distance_time(
            position, obstacle, self.config.time_horizon
        )

        time_weight = self._compute_time_dependent_weight(
            min_distance_time, current_distance, obstacle
        )

        if current_distance < self.config.min_distance:
            force_direction = safe_normalize_with_default(current_relative_pos, default_value=np.array([1.0, 0.0]))
            force_magnitude = self.config.k_repulsion * 100.0
        else:
            effective_radius = obstacle.radius + self.config.safety_margin

            if current_distance <= effective_radius:
                force_magnitude = self.config.k_repulsion * (
                    1.0 / current_distance - 1.0 / effective_radius
                )
            else:
                decay_factor = np.exp(-(current_distance - effective_radius) / effective_radius)
                force_magnitude = self.config.k_repulsion * 0.1 * decay_factor

            force_direction = safe_normalize(current_relative_pos)

        repulsive_force = force_magnitude * time_weight * force_direction
        return repulsive_force

    def _compute_minimum_distance_time(self, position: Position,
                                     obstacle: DynamicObstacle,
                                     horizon: float) -> float:
        """最小距離到達時間の計算"""
        rel_pos = position.to_array() - obstacle.position.to_array()
        rel_vel = -obstacle.velocity.to_array()

        rel_speed = np.linalg.norm(rel_vel)
        if rel_speed < 0.1:
            return 0.0

        t_min = -np.dot(rel_pos, rel_vel) / (rel_speed ** 2)
        return max(0.0, min(t_min, horizon))

    def _compute_time_dependent_weight(self, min_distance_time: float,
                                     current_distance: float,
                                     obstacle: DynamicObstacle) -> float:
        """時間依存重み計算"""
        base_weight = 1.0
        time_urgency = np.exp(-min_distance_time / 1.0)
        distance_urgency = np.exp(-current_distance / obstacle.radius)
        total_weight = base_weight * (1.0 + time_urgency + distance_urgency)
        return min(total_weight, 10.0)

    def _compute_time_correction(self, position: Position, time: float,
                               obstacles: List[DynamicObstacle],
                               goal: Goal) -> np.ndarray:
        """時間依存補正項の計算"""
        correction = np.zeros(2)
        prediction_times = [0.5, 1.0, 2.0]

        for pred_time in prediction_times:
            time_correction = self._compute_prediction_correction(
                position, obstacles, pred_time, goal
            )
            time_weight = np.exp(-pred_time / 2.0)
            correction += time_weight * time_correction

        return correction * 0.2
    
    def _compute_path_following_force(self, position: Position, 
                                    planned_path: Path) -> np.ndarray:
        """A*経路追従力計算"""
        if not planned_path.points:
            return np.zeros(2)
        
        # 1. 現在位置に最も近い経路点を見つける
        closest_point_idx = self._find_closest_path_point(position, planned_path)
        
        # 2. 先読み距離分だけ前の点を目標とする
        target_point_idx = self._find_lookahead_point(
            position, planned_path, closest_point_idx
        )
        
        if target_point_idx >= len(planned_path.points):
            target_point_idx = len(planned_path.points) - 1
        
        target_position = planned_path.points[target_point_idx].position
        
        # 3. 目標点への方向ベクトル計算
        direction_vector = target_position.to_array() - position.to_array()
        distance = np.linalg.norm(direction_vector)
        
        if distance < self.config.min_distance:
            return np.zeros(2)
        
        # 4. 経路からの横方向距離に基づく補正
        cross_track_error = self._compute_cross_track_error(
            position, planned_path, closest_point_idx
        )
        
        # 5. 経路追従力の計算
        # 前進成分（経路に沿った方向）
        forward_force = self.config.k_path_attraction * safe_normalize(direction_vector)
        
        # 横方向補正成分（経路に戻る方向）
        lateral_correction = -0.5 * self.config.k_path_attraction * cross_track_error
        
        return forward_force + lateral_correction
    
    def _compute_adaptive_goal_force(self, position: Position, goal: Goal, 
                                   planned_path: Path) -> np.ndarray:
        """経路考慮型ゴール引力計算"""
        # 経路の最終点に近い場合のみゴール引力を適用
        if not planned_path.points:
            return self._compute_attractive_force(position, goal)
        
        final_path_point = planned_path.points[-1].position
        distance_to_final = position.distance_to(final_path_point)
        
        # 経路終端付近でのみゴール引力を強化
        if distance_to_final < self.config.lookahead_distance * 2:
            goal_force = self._compute_attractive_force(position, goal)
            # 距離に応じて重みを調整
            weight = min(1.0, distance_to_final / self.config.lookahead_distance)
            return goal_force * weight
        else:
            # 経路追従を優先
            return np.zeros(2)
    
    def _find_closest_path_point(self, position: Position, 
                               planned_path: Path) -> int:
        """現在位置に最も近い経路点のインデックスを見つける"""
        min_distance = float('inf')
        closest_idx = 0
        
        for i, path_point in enumerate(planned_path.points):
            distance = position.distance_to(path_point.position)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        
        return closest_idx
    
    def _find_lookahead_point(self, position: Position, planned_path: Path, 
                            start_idx: int) -> int:
        """先読み距離に基づく目標点インデックスを見つける"""
        accumulated_distance = 0.0
        current_pos = position
        
        for i in range(start_idx, len(planned_path.points)):
            path_point = planned_path.points[i]
            segment_distance = current_pos.distance_to(path_point.position)
            accumulated_distance += segment_distance
            
            if accumulated_distance >= self.config.lookahead_distance:
                return i
            
            current_pos = path_point.position
        
        # 経路終端に到達
        return len(planned_path.points) - 1
    
    def _compute_cross_track_error(self, position: Position, planned_path: Path, 
                                 closest_idx: int) -> np.ndarray:
        """経路からの横方向誤差ベクトル計算"""
        if closest_idx >= len(planned_path.points) - 1:
            return np.zeros(2)
        
        # 現在の経路セグメント
        current_point = planned_path.points[closest_idx].position
        next_point = planned_path.points[closest_idx + 1].position
        
        # 経路方向ベクトル
        path_direction = next_point.to_array() - current_point.to_array()
        path_length = np.linalg.norm(path_direction)
        
        if path_length < self.config.min_distance:
            return np.zeros(2)
        
        path_unit = path_direction / path_length
        
        # 現在位置から経路点への ベクトル
        position_vector = position.to_array() - current_point.to_array()
        
        # 経路に垂直な成分（横方向誤差）
        cross_track_distance = np.dot(position_vector, np.array([-path_unit[1], path_unit[0]]))
        cross_track_vector = np.array([-path_unit[1], path_unit[0]]) * cross_track_distance
        
        return cross_track_vector

    def _compute_prediction_correction(self, position: Position,
                                     obstacles: List[DynamicObstacle],
                                     prediction_time: float,
                                     goal: Goal) -> np.ndarray:
        """指定時間後の予測に基づく補正ベクトル計算"""
        correction = np.zeros(2)

        for obstacle in obstacles:
            predicted_pos = obstacle.predict_position(prediction_time)
            predicted_relative = position.to_array() - predicted_pos.to_array()
            predicted_distance = np.linalg.norm(predicted_relative)

            if predicted_distance < self.config.influence_radius:
                avoidance_strength = (self.config.influence_radius - predicted_distance) / self.config.influence_radius
                avoidance_direction = safe_normalize(predicted_relative)
                correction += avoidance_strength * avoidance_direction * 0.5

        return correction

# ============================================================================
# Velocity Obstacles計算クラス
# ============================================================================

class VelocityObstacleCalculator:
    """Velocity Obstacle計算器"""

    def __init__(self, config: TVVFVOConfig):
        self.config = config

    def compute_vo_set(self, robot_state: RobotState,
                      obstacles: List[DynamicObstacle],
                      time_horizon: float) -> List[Dict]:
        """VO集合計算"""
        vo_cones = []

        for obstacle in obstacles:
            vo_cone = self._compute_single_vo(robot_state, obstacle, time_horizon)
            if vo_cone is not None:
                vo_cones.append(vo_cone)

        return vo_cones

    def _compute_single_vo(self, robot_state: RobotState,
                          obstacle: DynamicObstacle,
                          time_horizon: float) -> Optional[Dict]:
        """単一障害物に対するVO錐体計算"""
        rel_pos = obstacle.position.to_array() - robot_state.position.to_array()
        rel_vel = obstacle.velocity.to_array()
        rel_distance = np.linalg.norm(rel_pos)

        max_relative_speed = robot_state.max_velocity + obstacle.velocity.magnitude()
        max_influence_distance = max_relative_speed * time_horizon

        if rel_distance > max_influence_distance:
            return None

        expanded_radius = obstacle.radius + robot_state.radius + self.config.safety_margin
        cone_vertex = rel_vel

        if rel_distance <= expanded_radius:
            return {
                'type': 'full_circle',
                'center': cone_vertex,
                'radius': robot_state.max_velocity,
                'obstacle_id': obstacle.id
            }

        sin_theta = expanded_radius / rel_distance
        cos_theta = np.sqrt(1 - sin_theta * sin_theta)
        rel_pos_norm = safe_normalize(rel_pos)

        tangent_right = np.array([
            cos_theta * rel_pos_norm[0] - sin_theta * rel_pos_norm[1],
            sin_theta * rel_pos_norm[0] + cos_theta * rel_pos_norm[1]
        ])

        tangent_left = np.array([
            cos_theta * rel_pos_norm[0] + sin_theta * rel_pos_norm[1],
            -sin_theta * rel_pos_norm[0] + cos_theta * rel_pos_norm[1]
        ])

        cone_right = cone_vertex + tangent_right * robot_state.max_velocity
        cone_left = cone_vertex + tangent_left * robot_state.max_velocity

        return {
            'type': 'cone',
            'cone_vertex': cone_vertex,
            'cone_left': cone_left,
            'cone_right': cone_right,
            'obstacle_id': obstacle.id,
            'tangent_left': tangent_left,
            'tangent_right': tangent_right
        }

    def is_velocity_in_vo(self, velocity: np.ndarray, vo_cone: Dict) -> bool:
        """指定速度がVO錐体内にあるかチェック"""
        if vo_cone['type'] == 'full_circle':
            relative_vel = velocity - vo_cone['center']
            return np.linalg.norm(relative_vel) <= vo_cone['radius']
        elif vo_cone['type'] == 'cone':
            relative_vel = velocity - vo_cone['cone_vertex']
            cross_left = np.cross(vo_cone['tangent_left'], relative_vel)
            cross_right = np.cross(relative_vel, vo_cone['tangent_right'])
            return cross_left >= 0 and cross_right >= 0
        return False

    def get_vo_free_velocities(self, robot_state: RobotState,
                              vo_cones: List[Dict],
                              resolution: float = None) -> List[np.ndarray]:
        """VO制約を満たす実行可能速度のサンプリング"""
        if resolution is None:
            resolution = self.config.vo_resolution

        feasible_velocities = []
        max_vel = robot_state.max_velocity
        vel_range = np.arange(-max_vel, max_vel + resolution, resolution)

        for vx in vel_range:
            for vy in vel_range:
                velocity = np.array([vx, vy])

                if np.linalg.norm(velocity) > max_vel:
                    continue

                is_feasible = True
                for vo_cone in vo_cones:
                    if self.is_velocity_in_vo(velocity, vo_cone):
                        is_feasible = False
                        break

                if is_feasible:
                    feasible_velocities.append(velocity)

        return feasible_velocities

# ============================================================================
# 実行可能速度選択クラス
# ============================================================================

class FeasibleVelocitySelector:
    """実行可能速度選択器"""

    def __init__(self, config: TVVFVOConfig):
        self.config = config

    def select_feasible_velocity(self, tvvf_vector: np.ndarray,
                                vo_cones: List[Dict],
                                robot_state: RobotState) -> Velocity:
        """TVVFベクトルとVO制約から実行可能速度選択"""
        candidate_velocities = self._generate_candidate_velocities(
            tvvf_vector, vo_cones, robot_state
        )

        if not candidate_velocities:
            return Velocity(0.0, 0.0)

        best_velocity = self._select_optimal_velocity(
            candidate_velocities, tvvf_vector, vo_cones, robot_state
        )

        return Velocity(best_velocity[0], best_velocity[1])

    def _generate_candidate_velocities(self, tvvf_vector: np.ndarray,
                                     vo_cones: List[Dict],
                                     robot_state: RobotState) -> List[np.ndarray]:
        """候補速度の生成"""
        candidates = []

        # TVVF方向の速度
        tvvf_magnitude = np.linalg.norm(tvvf_vector)
        if tvvf_magnitude > 0:
            tvvf_direction = tvvf_vector / tvvf_magnitude
            for speed_ratio in [0.3, 0.6, 0.8, 1.0]:
                speed = min(robot_state.max_velocity * speed_ratio, tvvf_magnitude)
                candidate = tvvf_direction * speed

                if self._is_velocity_feasible(candidate, vo_cones):
                    candidates.append(candidate)

        # 基本方向の速度
        basic_directions = [
            np.array([1.0, 0.0]), np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]), np.array([0.0, -1.0]),
            np.array([0.707, 0.707]), np.array([-0.707, 0.707]),
            np.array([0.707, -0.707]), np.array([-0.707, -0.707])
        ]

        for direction in basic_directions:
            for speed_ratio in [0.5, 1.0]:
                speed = robot_state.max_velocity * speed_ratio
                candidate = direction * speed

                if (np.linalg.norm(candidate) <= robot_state.max_velocity and
                    self._is_velocity_feasible(candidate, vo_cones)):
                    candidates.append(candidate)

        # 停止速度
        if self._is_velocity_feasible(np.array([0.0, 0.0]), vo_cones):
            candidates.append(np.array([0.0, 0.0]))

        return candidates

    def _is_velocity_feasible(self, velocity: np.ndarray, vo_cones: List[Dict]) -> bool:
        """速度がVO制約を満たすかチェック"""
        vo_calc = VelocityObstacleCalculator(self.config)

        for vo_cone in vo_cones:
            if vo_calc.is_velocity_in_vo(velocity, vo_cone):
                return False
        return True

    def _select_optimal_velocity(self, candidates: List[np.ndarray],
                               tvvf_vector: np.ndarray,
                               vo_cones: List[Dict],
                               robot_state: RobotState) -> np.ndarray:
        """最適速度の選択"""
        best_velocity = candidates[0]
        best_score = float('-inf')

        tvvf_magnitude = np.linalg.norm(tvvf_vector)
        tvvf_direction = safe_normalize(tvvf_vector) if tvvf_magnitude > 0 else np.zeros(2)

        for candidate in candidates:
            score = self._compute_velocity_score(
                candidate, tvvf_direction, tvvf_magnitude, vo_cones, robot_state
            )

            if score > best_score:
                best_score = score
                best_velocity = candidate

        return best_velocity

    def _compute_velocity_score(self, velocity: np.ndarray,
                              tvvf_direction: np.ndarray,
                              tvvf_magnitude: float,
                              vo_cones: List[Dict],
                              robot_state: RobotState) -> float:
        """速度の評価スコア計算"""
        score = 0.0

        # TVVF方向類似性
        vel_magnitude = np.linalg.norm(velocity)
        if vel_magnitude > 0 and tvvf_magnitude > 0:
            vel_direction = velocity / vel_magnitude
            direction_similarity = np.dot(vel_direction, tvvf_direction)
            score += self.config.direction_weight * direction_similarity

        # VO安全マージン
        safety_score = self._compute_safety_score(velocity, vo_cones)
        score += self.config.safety_weight * safety_score

        # エネルギー効率
        efficiency_score = 1.0 - (vel_magnitude / robot_state.max_velocity)
        score += self.config.efficiency_weight * efficiency_score

        # 速度連続性
        current_vel = robot_state.velocity.to_array()
        velocity_change = np.linalg.norm(velocity - current_vel)
        continuity_score = 1.0 / (1.0 + velocity_change)
        score += 0.5 * continuity_score

        return score

    def _compute_safety_score(self, velocity: np.ndarray, vo_cones: List[Dict]) -> float:
        """VO安全マージンスコア計算"""
        if not vo_cones:
            return 1.0

        min_distance = float('inf')

        for vo_cone in vo_cones:
            if vo_cone['type'] == 'cone':
                vertex = vo_cone['cone_vertex']
                rel_vel = velocity - vertex

                left_tangent = vo_cone['tangent_left']
                right_tangent = vo_cone['tangent_right']

                left_distance = abs(np.cross(left_tangent, rel_vel))
                right_distance = abs(np.cross(right_tangent, rel_vel))
                boundary_distance = min(left_distance, right_distance)

                min_distance = min(min_distance, boundary_distance)

        return min(1.0, min_distance / 0.5)

# ============================================================================
# 統合制御器クラス
# ============================================================================

class TVVFVOController:
    """TVVF + VO統合制御器"""

    def __init__(self, config: TVVFVOConfig):
        self.config = config
        self.tvvf_generator = TVVFGenerator(config)
        self.vo_calculator = VelocityObstacleCalculator(config)
        self.velocity_selector = FeasibleVelocitySelector(config)

        # 統計情報
        self.stats = {
            'computation_time': 0.0,
            'tvvf_time': 0.0,
            'vo_time': 0.0,
            'selection_time': 0.0,
            'num_vo_cones': 0,
            'num_candidates': 0,
            'safety_margin': 0.0
        }

    def update(self, robot_state: RobotState,
              obstacles: List[DynamicObstacle],
              goal: Goal, 
              planned_path: Optional[Path] = None) -> ControlOutput:
        """制御更新関数"""
        start_time = time.time()

        try:
            # A*経路の存在チェック
            if planned_path is None or not planned_path.points:
                # 経路がない場合は停止コマンドを返す
                return ControlOutput(
                    velocity_command=Velocity(0.0, 0.0),
                    angular_velocity=0.0,
                    execution_time=0.01,
                    safety_margin=0.0
                )
            
            # TVVF計算（A*経路統合版）
            tvvf_start = time.time()
            tvvf_vector = self.tvvf_generator.compute_vector(
                robot_state.position, start_time, goal, obstacles, planned_path
            )
            self.stats['tvvf_time'] = (time.time() - tvvf_start) * 1000

            # VO制約計算
            vo_start = time.time()
            vo_cones = self.vo_calculator.compute_vo_set(
                robot_state, obstacles, self.config.time_horizon
            )
            self.stats['vo_time'] = (time.time() - vo_start) * 1000
            self.stats['num_vo_cones'] = len(vo_cones)

            # 実行可能速度選択
            selection_start = time.time()
            selected_velocity = self.velocity_selector.select_feasible_velocity(
                tvvf_vector, vo_cones, robot_state
            )
            self.stats['selection_time'] = (time.time() - selection_start) * 1000

            # 安全性評価
            safety_margin = self._compute_safety_margin(robot_state, obstacles)
            self.stats['safety_margin'] = safety_margin

            # 制御コマンド生成
            control_output = self._generate_control_output(
                selected_velocity, robot_state, safety_margin
            )

            self.stats['computation_time'] = (time.time() - start_time) * 1000

            return control_output

        except Exception as e:
            print(f"制御計算エラー: {e}")
            return ControlOutput(
                velocity_command=Velocity(0.0, 0.0),
                angular_velocity=0.0,
                execution_time=self.config.max_computation_time,
                safety_margin=0.0
            )

    def _compute_safety_margin(self, robot_state: RobotState,
                             obstacles: List[DynamicObstacle]) -> float:
        """現在の安全マージン計算"""
        if not obstacles:
            return float('inf')

        min_distance = float('inf')
        for obstacle in obstacles:
            distance = robot_state.position.distance_to(obstacle.position)
            effective_distance = distance - robot_state.radius - obstacle.radius
            min_distance = min(min_distance, effective_distance)

        return max(0.0, min_distance)

    def _generate_control_output(self, selected_velocity: Velocity,
                               robot_state: RobotState,
                               safety_margin: float) -> ControlOutput:
        """制御コマンド生成"""
        target_angle = np.arctan2(selected_velocity.vy, selected_velocity.vx)
        angle_diff = target_angle - robot_state.orientation

        # 角度差の正規化
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        angular_velocity = angle_diff * 2.0

        return ControlOutput(
            velocity_command=selected_velocity,
            angular_velocity=angular_velocity,
            execution_time=self.config.max_computation_time,
            safety_margin=safety_margin
        )

    def get_stats(self) -> Dict:
        """統計情報取得"""
        return self.stats.copy()

# ============================================================================
# A*経路計画クラス
# ============================================================================

class AStarNode:
    """A*アルゴリズム用ノードクラス"""
    
    def __init__(self, position: Tuple[int, int], g_cost: float = 0.0, h_cost: float = 0.0, parent=None):
        self.position = position  # (x, y) グリッド座標
        self.g_cost = g_cost      # スタートからのコスト
        self.h_cost = h_cost      # ゴールまでのヒューリスティックコスト
        self.f_cost = g_cost + h_cost  # 総コスト
        self.parent = parent      # 親ノード
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        return self.position == other.position

class AStarPathPlanner:
    """A*アルゴリズムによる経路計画クラス"""
    
    def __init__(self, occupancy_grid, resolution: float, origin: Position, 
                 wall_clearance_distance: float = 0.8):
        """
        Args:
            occupancy_grid: OccupancyGridメッセージ
            resolution: グリッドの解像度 [m/cell]
            origin: グリッドの原点座標
            wall_clearance_distance: 壁からの最小距離 [m]
        """
        self.width = occupancy_grid.info.width
        self.height = occupancy_grid.info.height
        self.resolution = resolution
        self.origin = origin
        self.wall_clearance_distance = wall_clearance_distance
        
        # 占有格子データを2D配列に変換
        self.grid = np.array(occupancy_grid.data).reshape((self.height, self.width))
        
        # 占有閾値（0-100で100が完全占有）
        self.occupied_threshold = 65
        self.free_threshold = 25
        
        # 壁からの距離を考慮した膨張マップを作成
        self.inflated_grid = self._create_inflated_grid()
    
    def world_to_grid(self, world_pos: Position) -> Tuple[int, int]:
        """世界座標をグリッド座標に変換"""
        grid_x = int((world_pos.x - self.origin.x) / self.resolution)
        grid_y = int((world_pos.y - self.origin.y) / self.resolution)
        return (grid_x, grid_y)
    
    def grid_to_world(self, grid_pos: Tuple[int, int]) -> Position:
        """グリッド座標を世界座標に変換"""
        world_x = grid_pos[0] * self.resolution + self.origin.x
        world_y = grid_pos[1] * self.resolution + self.origin.y
        return Position(world_x, world_y)
    
    def _create_inflated_grid(self) -> np.ndarray:
        """壁からの距離を考慮した膨張マップを作成"""
        inflated_grid = np.copy(self.grid)
        
        # 膨張半径をグリッドセルで計算
        inflation_cells = int(self.wall_clearance_distance / self.resolution) + 1
        
        # 元の占有セルを特定
        occupied_cells = np.where(self.grid >= self.occupied_threshold)
        
        # 各占有セルの周囲を膨張
        for y, x in zip(occupied_cells[0], occupied_cells[1]):
            for dy in range(-inflation_cells, inflation_cells + 1):
                for dx in range(-inflation_cells, inflation_cells + 1):
                    new_y, new_x = y + dy, x + dx
                    
                    # 境界チェック
                    if (0 <= new_x < self.width and 0 <= new_y < self.height):
                        # ユークリッド距離で膨張範囲を決定
                        distance = math.sqrt(dx**2 + dy**2) * self.resolution
                        if distance <= self.wall_clearance_distance:
                            # 元が自由空間の場合のみ膨張値を設定
                            if inflated_grid[new_y, new_x] < self.occupied_threshold:
                                inflated_grid[new_y, new_x] = self.occupied_threshold
        
        return inflated_grid

    def is_valid_position(self, grid_pos: Tuple[int, int]) -> bool:
        """グリッド位置が有効かチェック（膨張マップ使用）"""
        x, y = grid_pos
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        
        # 膨張マップで占有状態をチェック
        cell_value = self.inflated_grid[y, x]
        return cell_value >= 0 and cell_value < self.occupied_threshold
    
    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """8近傍の有効な隣接セルを取得"""
        x, y = position
        neighbors = []
        
        # 8方向の移動（対角線も含む）
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if self.is_valid_position((new_x, new_y)):
                neighbors.append((new_x, new_y))
        
        return neighbors
    
    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """ユークリッド距離ヒューリスティック"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_movement_cost(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        """移動コスト計算（対角線移動は√2倍）"""
        dx = abs(to_pos[0] - from_pos[0])
        dy = abs(to_pos[1] - from_pos[1])
        
        if dx == 1 and dy == 1:
            return math.sqrt(2)  # 対角線移動
        else:
            return 1.0  # 直線移動
    
    def plan_path(self, start_pos: Position, goal_pos: Position) -> Optional[Path]:
        """A*アルゴリズムによる経路計画"""
        # 世界座標をグリッド座標に変換
        start_grid = self.world_to_grid(start_pos)
        goal_grid = self.world_to_grid(goal_pos)
        
        # 開始・終了位置の有効性チェック
        if not self.is_valid_position(start_grid):
            print(f"開始位置が無効: {start_grid}")
            return None
        
        if not self.is_valid_position(goal_grid):
            print(f"目標位置が無効: {goal_grid}")
            return None
        
        # A*アルゴリズム
        open_set = []
        closed_set = set()
        
        # 開始ノード
        start_node = AStarNode(
            start_grid, 
            g_cost=0.0, 
            h_cost=self.heuristic(start_grid, goal_grid)
        )
        heapq.heappush(open_set, start_node)
        
        # ノード探索用辞書
        open_dict = {start_grid: start_node}
        
        while open_set:
            # 最小コストノードを取得
            current_node = heapq.heappop(open_set)
            current_pos = current_node.position
            
            # 辞書からも削除
            if current_pos in open_dict:
                del open_dict[current_pos]
            
            # ゴール到達チェック
            if current_pos == goal_grid:
                return self._reconstruct_path(current_node)
            
            # クローズドセットに追加
            closed_set.add(current_pos)
            
            # 隣接ノードを探索
            for neighbor_pos in self.get_neighbors(current_pos):
                if neighbor_pos in closed_set:
                    continue
                
                # 移動コスト計算
                movement_cost = self.get_movement_cost(current_pos, neighbor_pos)
                tentative_g_cost = current_node.g_cost + movement_cost
                
                # 既存ノードのチェック
                if neighbor_pos in open_dict:
                    neighbor_node = open_dict[neighbor_pos]
                    if tentative_g_cost < neighbor_node.g_cost:
                        # より良いパスを発見
                        neighbor_node.g_cost = tentative_g_cost
                        neighbor_node.f_cost = tentative_g_cost + neighbor_node.h_cost
                        neighbor_node.parent = current_node
                        heapq.heapify(open_set)  # ヒープを再構築
                else:
                    # 新しいノードを作成
                    h_cost = self.heuristic(neighbor_pos, goal_grid)
                    neighbor_node = AStarNode(
                        neighbor_pos, 
                        g_cost=tentative_g_cost, 
                        h_cost=h_cost, 
                        parent=current_node
                    )
                    heapq.heappush(open_set, neighbor_node)
                    open_dict[neighbor_pos] = neighbor_node
        
        # 経路が見つからない
        print("経路が見つかりませんでした")
        return None
    
    def _reconstruct_path(self, goal_node: AStarNode) -> Path:
        """ゴールノードから経路を再構築"""
        path = Path()
        current_node = goal_node
        grid_path = []
        
        # 逆順にたどって経路を構築
        while current_node is not None:
            grid_path.append(current_node.position)
            current_node = current_node.parent
        
        # 正順に変換して世界座標に変換
        grid_path.reverse()
        
        for i, grid_pos in enumerate(grid_path):
            world_pos = self.grid_to_world(grid_pos)
            # 移動コストを計算（簡易版）
            if i > 0:
                prev_world_pos = self.grid_to_world(grid_path[i-1])
                cost = world_pos.distance_to(prev_world_pos)
            else:
                cost = 0.0
            
            path.add_point(world_pos, cost)
        
        return path