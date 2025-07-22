#!/usr/bin/env python3
"""
TVVF-VO ROS2 Launch File
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # パッケージパス取得
    pkg_share = FindPackageShare('tvvf_vo')

    # Launch引数の定義
    declare_config_file = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([pkg_share, 'config', 'tvvf_vo_params.yaml']),
        description='パラメータファイルのパス'
    )

    declare_enable_visualization = DeclareLaunchArgument(
        'enable_visualization',
        default_value='true',
        description='可視化マーカーを有効にするかどうか'
    )



    declare_max_velocity = DeclareLaunchArgument(
        'max_velocity',
        default_value='1.5',
        description='最大速度 [m/s]'
    )

    declare_robot_radius = DeclareLaunchArgument(
        'robot_radius',
        default_value='0.3',
        description='ロボット半径 [m]'
    )

    # TVVF-VOノード
    tvvf_vo_node = Node(
        package='tvvf_vo',
        executable='tvvf_vo_node',
        name='tvvf_vo_node',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'enable_visualization': LaunchConfiguration('enable_visualization'),
                'max_velocity': LaunchConfiguration('max_velocity'),
                'robot_radius': LaunchConfiguration('robot_radius'),
            }
        ],
        remappings=[
            ('odom', 'odom'),
            ('scan', 'scan'),
            ('goal_pose', 'goal_pose'),
            ('cmd_vel', 'cmd_vel'),
            ('tvvf_vo_markers', 'tvvf_vo_markers'),
            ('tvvf_vo_debug', 'tvvf_vo_debug'),
        ]
    )

    return LaunchDescription([
        declare_config_file,
        declare_enable_visualization,
        declare_max_velocity,
        declare_robot_radius,
        tvvf_vo_node,
    ])