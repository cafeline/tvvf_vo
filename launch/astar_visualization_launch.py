#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """A* Path Planning Test with RViz launch file"""

    # Package path
    pkg_tvvf_vo = FindPackageShare('tvvf_vo')

    # Launch arguments
    map_topic_arg = DeclareLaunchArgument(
        'map_topic',
        default_value='map',
        description='Topic name for occupancy grid map'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution([
            pkg_tvvf_vo,
            'config',
            'astar_visualization.rviz'
        ]),
        description='RViz configuration file path'
    )

    # TVVF-VO Node with A* path planning
    tvvf_vo_node = Node(
        package='tvvf_vo',
        executable='tvvf_vo_node',
        name='tvvf_vo_node',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                pkg_tvvf_vo,
                'config',
                'tvvf_vo_params.yaml'
            ]),
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
        remappings=[
            ('map', LaunchConfiguration('map_topic')),
        ]
    )

    # RViz Node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
        output='screen'
    )

    return LaunchDescription([
        map_topic_arg,
        use_sim_time_arg,
        rviz_config_arg,
        tvvf_vo_node,
        rviz_node,
    ])
