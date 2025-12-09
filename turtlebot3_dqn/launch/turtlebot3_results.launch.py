#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='turtlebot3_dqn',
            executable='result_graph',
            name='result_graph',
            output='screen',
        ),
        Node(
            package='turtlebot3_dqn',
            executable='action_graph',
            name='action_graph',
            output='screen',
        ),
    ])
