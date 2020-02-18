#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Ryan Shim

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
import os


def generate_launch_description():
    stage = LaunchConfiguration('stage', default='1')

    saved_model_file_name = 'stage1_'
    saved_model = os.path.join(
        get_package_share_directory('turtlebot3_dqn'),
        'save_model',
        saved_model_file_name)

    gazebo_model = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'models/turtlebot3_square/goal_box/model.sdf')

    return LaunchDescription([
        Node(
            package='turtlebot3_dqn',
            node_executable='turtlebot3_dqn',
            node_name='turtlebot3_dqn',
            output='screen',
            arguments=[saved_model]),

        Node(
            package='turtlebot3_dqn',
            node_executable='dqn_environment',
            node_name='dqn_environment',
            output='screen',
            arguments=[stage]),

        Node(
            package='turtlebot3_dqn',
            node_executable='turtlebot3_graph',
            node_name='action_graph',
            output='screen'),

        Node(
            package='turtlebot3_dqn',
            node_executable='turtlebot3_graph',
            node_name='result_graph',
            output='screen'),
    ])
