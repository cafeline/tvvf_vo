# TVVF-VO ROS2パッケージ

Time-Varying Vector Field (TVVF) と Velocity Obstacles (VO) を統合した動的環境ナビゲーションシステムのROS2実装です。

## 概要

このパッケージは、動的障害物環境でのロボットナビゲーションを実現するために、以下の技術を統合しています：

- **Time-Varying Vector Field (TVVF)**: 時間依存ベクトル場による経路計画
- **Velocity Obstacles (VO)**: 動的障害物回避のための速度制約
- **リアルタイム制御**: 20Hz制御ループでの高速計算
- **ROS2統合**: ROS2 Humbleとの完全統合

## 特徴

✅ **動的障害物回避**: 移動する障害物を予測して回避
✅ **数値安定性**: ゼロ除算や特異点を回避する安全な実装
✅ **リアルタイム性能**: 50ms以内での計算完了
✅ **可視化**: RVizでの詳細な可視化サポート
✅ **パラメータ調整**: ROS2パラメータによる柔軟な設定

## インストール

### 前提条件
- ROS2 Humble
- Python 3.8+
- numpy
- tf_transformations

### ビルド
```bash
cd ~/your_ros2_workspace
colcon build --packages-select tvvf_vo
source install/setup.bash
```

## 使用方法

### 基本起動
```bash
# ノード単体起動
ros2 run tvvf_vo tvvf_vo_node

# launchファイルで起動（推奨）
ros2 launch tvvf_vo tvvf_vo_launch.py
```

### RViz2との統合使用
```bash
# 1. TVVF-VOノードを起動
ros2 launch tvvf_vo tvvf_vo_launch.py

# 2. RViz2を起動
ros2 run rviz2 rviz2

# 3. RViz2での設定
# - Fixed Frame: 'map'
# - Add → MarkerArray → Topic: '/tvvf_vo_markers'
# - Add → LaserScan → Topic: '/scan' (利用可能な場合)

# 4. ゴールの設定
# - コマンドラインで /goal_pose トピックに目標を送信
# - または、RViz2でPublishツールを使用して /goal_pose に送信
```

### パラメータ付き起動
```bash
ros2 launch tvvf_vo tvvf_vo_launch.py \
    max_velocity:=2.0 \
    robot_radius:=0.4 \
    enable_debug:=true
```

### 設定ファイルを指定
```bash
ros2 launch tvvf_vo tvvf_vo_launch.py \
    config_file:=/path/to/your/params.yaml
```

## トピック

### 購読 (Subscribe)
- `/odom` (nav_msgs/Odometry): ロボットの位置・速度情報
- `/scan` (sensor_msgs/LaserScan): **動的障害物検出** - レーザースキャンからクラスタリングで障害物を検出
- `/goal_pose` (geometry_msgs/PoseStamped): 目標位置

### 配信 (Publish)
- `/cmd_vel` (geometry_msgs/Twist): 制御コマンド
- `/tvvf_vo_markers` (visualization_msgs/MarkerArray): 可視化マーカー
- `/tvvf_vo_debug` (visualization_msgs/Marker): デバッグ用マーカー

### TF変換
- `map` → `base_link`: ロボットの現在位置（tfから自動取得）
- 各フレームは `base_frame` と `global_frame` パラメータで設定可能

## パラメータ

### 主要パラメータ
```yaml
# TVVF関連
k_attraction: 1.2      # 引力場強度
k_repulsion: 2.5       # 斥力場強度
influence_radius: 3.0  # 障害物影響半径 [m]

# VO関連
time_horizon: 3.0      # VO時間ホライズン [s]
safety_margin: 0.25    # 安全マージン [m]

# ロボット関連
max_velocity: 1.5      # 最大速度 [m/s]
robot_radius: 0.3      # ロボット半径 [m]
```

詳細は `config/tvvf_vo_params.yaml` を参照してください。

## RVizでの可視化

RVizで以下のトピックを追加することで可視化できます：

```bash
# RViz起動
rviz2

# 追加するトピック：
# - /tvvf_vo_markers (MarkerArray): ロボット、目標、障害物
# - /tvvf_vo_debug (Marker): デバッグ情報
```

## 使用例

### Gazeboシミュレーション
```bash
# 1. Gazebo環境起動
ros2 launch your_gazebo_package your_world.launch.py

# 2. TVVF-VOナビゲーション起動
ros2 launch tvvf_vo tvvf_vo_launch.py

# 3. RViz2起動と目標設定
ros2 run rviz2 rviz2
# RViz2のPublishツールまたはコマンドラインで /goal_pose にゴールを送信

# または、コマンドラインから目標位置を設定
ros2 topic pub /goal_pose geometry_msgs/PoseStamped "{
  header: {frame_id: 'map'},
  pose: {
    position: {x: 5.0, y: 2.0, z: 0.0},
    orientation: {w: 1.0}
  }
}"
```

### 実機ロボット
```bash
# 1. ロボットドライバ起動
ros2 launch your_robot_package your_robot.launch.py

# 2. TVVF-VOナビゲーション起動（実機用設定）
ros2 launch tvvf_vo tvvf_vo_launch.py \
    config_file:=config/real_robot_params.yaml \
    max_velocity:=1.0
```

## トラブルシューティング

### 計算時間が遅い場合
```yaml
# パラメータ調整
vo_resolution: 0.2      # 解像度を下げる
influence_radius: 2.5   # 影響半径を小さくする
```

### 回避が保守的すぎる場合
```yaml
# パラメータ調整
safety_margin: 0.15     # 安全マージンを小さくする
k_repulsion: 1.5        # 斥力を弱くする
```

### デバッグ情報の有効化
```bash
ros2 param set /tvvf_vo_node enable_debug_output true
```

## アーキテクチャ

```
tvvf_vo/
├── tvvf_vo/
│   ├── tvvf_vo_core.py     # コアアルゴリズム
│   ├── tvvf_vo_node.py     # ROS2ノード実装
│   └── tvvf_vo.py          # エントリーポイント
├── launch/
│   └── tvvf_vo_launch.py   # Launchファイル
├── config/
│   └── tvvf_vo_params.yaml # パラメータファイル
└── README.md
```

## ライセンス

Apache-2.0

## 貢献

バグ報告や機能提案は Issues からお願いします。

## 参考文献

- Time-Varying Vector Fields for Robot Navigation
- Velocity Obstacles for Multi-Agent Systems
- ROS2 Navigation Stack