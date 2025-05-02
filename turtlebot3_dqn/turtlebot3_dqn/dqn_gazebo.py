#!/usr/bin/env python3
#################################################################################
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
#################################################################################
#
# # Authors: Ryan Shim, Gilbert, ChanHyeong Lee

import os
import random
import subprocess
import sys
import time

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node

from turtlebot3_msgs.srv import Goal


class GazeboInterface(Node):

    def __init__(self, stage_num):
        super().__init__('gazebo_interface')
        self.stage = int(stage_num)

        self.entity_name = 'goal_box'
        self.entity_pose_x = 0.5
        self.entity_pose_y = 0.0

        self.callback_group = MutuallyExclusiveCallbackGroup()
        self.initialize_env_service = self.create_service(
            Goal,
            'initialize_env',
            self.initialize_env_callback,
            callback_group=self.callback_group
        )
        self.task_succeed_service = self.create_service(
            Goal,
            'task_succeed',
            self.task_succeed_callback,
            callback_group=self.callback_group
        )
        self.task_failed_service = self.create_service(
            Goal,
            'task_failed',
            self.task_failed_callback,
            callback_group=self.callback_group
        )

    def spawn_entity(self, x: float, y: float, z: float = 0.0):
        service_name = '/world/dqn/create'
        package_share = get_package_share_directory('turtlebot3_gazebo')
        model_path = os.path.join(
            package_share, 'models', 'turtlebot3_dqn_world', 'goal_box', 'model.sdf'
        )

        req = (
            f'sdf_filename: "{model_path}, '
            f'name: "{self.entity_name}", '
            f'pose: {{ position: {{ x: {x}, y: {y}, z: {z} }} }}'
        )
        cmd = [
            'gz', 'service',
            '-s', service_name,
            '--reqtype', 'gz.msgs.EntityFactory',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000',
            '--req', req
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            print(f'[✓] Spawn Entity at ({x}, {y}, {z})')
        except subprocess.CalledProcessError:
            pass

    def delete_entity(self):
        service_name = '/world/dqn/remove'
        req = f'name: {self.entity_name}, type: 2'
        cmd = [
            'gz', 'service',
            '-s', service_name,
            '--reqtype', 'gz.msgs.Entity',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000',
            '--req', req
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            print('[✓] Delete Entity')
        except subprocess.CalledProcessError:
            pass

    def reset_burger(self):
        service_name_delete = '/world/dqn/remove'
        req_delete = 'name: "burger", type: 2'
        cmd_delete = [
            'gz', 'service',
            '-s', service_name_delete,
            '--reqtype', 'gz.msgs.Entity',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000',
            '--req', req_delete
        ]
        try:
            subprocess.run(cmd_delete, check=True, stdout=subprocess.DEVNULL)
            print('[✓] Delete Burger')
        except subprocess.CalledProcessError:
            pass
        time.sleep(0.2)
        service_name_spawn = '/world/dqn/create'
        package_share = get_package_share_directory('turtlebot3_gazebo')
        model_path = os.path.join(package_share, 'models', 'turtlebot3_burger', 'model.sdf')
        req_spawn = (
            f'sdf_filename: "{model_path}", '
            f'name: "burger", '
            f'pose: {{ position: {{ x: 0.0, y: 0.0, z: 0.0 }} }}'
        )
        cmd_spawn = [
            'gz', 'service',
            '-s', service_name_spawn,
            '--reqtype', 'gz.msgs.EntityFactory',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000',
            '--req', req_spawn
        ]
        try:
            subprocess.run(cmd_spawn, check=True, stdout=subprocess.DEVNULL)
            print('[✓] Spawn Burger')
        except subprocess.CalledProcessError:
            pass

    def task_succeed_callback(self, request, response):
        self.delete_entity()
        time.sleep(0.2)
        self.generate_goal_pose()
        time.sleep(0.2)
        self.spawn_entity(self.entity_pose_x, self.entity_pose_y)
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        return response

    def task_failed_callback(self, request, response):
        self.delete_entity()
        time.sleep(0.2)
        self.reset_burger()
        time.sleep(0.2)
        self.generate_goal_pose()
        time.sleep(0.2)
        self.spawn_entity(self.entity_pose_x, self.entity_pose_y)
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        return response

    def initialize_env_callback(self, request, response):
        self.delete_entity()
        time.sleep(0.2)
        self.reset_burger()
        time.sleep(0.2)
        self.spawn_entity(self.entity_pose_x, self.entity_pose_y)
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        return response

    def generate_goal_pose(self):
        if self.stage != 4:
            self.entity_pose_x = random.randrange(-23, 23) / 10
            self.entity_pose_y = random.randrange(-23, 23) / 10
        else:
            goal_pose_list = [
                [1.0, 0.0], [2.0, -1.5], [0.0, -2.0], [2.0, 2.0], [0.8, 2.0], [-1.9, 1.9],
                [-1.9, 0.2], [-1.9, -0.5], [-2.0, -2.0], [-0.5, -1.0], [-0.5, 2.0], [2.0, -0.5]
            ]
            rand_index = random.randint(0, 11)
            self.entity_pose_x = goal_pose_list[rand_index][0]
            self.entity_pose_y = goal_pose_list[rand_index][1]


def main(args=None):
    rclpy.init(args=sys.argv)
    stage_num = sys.argv[1] if len(sys.argv) > 1 else '1'
    gazebo_interface = GazeboInterface(stage_num)
    try:
        while rclpy.ok():
            rclpy.spin_once(gazebo_interface, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        gazebo_interface.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
