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
# Authors: Ryan Shim, Gilbert

import os
import random
import sys
import time

from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from std_srvs.srv import Empty

from turtlebot3_msgs.srv import Goal


class GazeboInterface(Node):

    def __init__(self, stage):
        super().__init__('gazebo_interface')

        self.stage = int(stage)

        self.entity_name = 'Goal'
        self.entity = None
        self.open_entity()

        self.entity_pose_x = 0.5
        self.entity_pose_y = 0.0

        self.delete_entity_client = self.create_client(DeleteEntity, 'delete_entity')
        self.spawn_entity_client = self.create_client(SpawnEntity, 'spawn_entity')
        self.reset_simulation_client = self.create_client(Empty, 'reset_simulation')

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

    def open_entity(self):
        try:
            package_share = get_package_share_directory('turtlebot3_gazebo')
            model_path = os.path.join(
                package_share, 'models', 'turtlebot3_dqn_world', 'goal_box', 'model.sdf'
            )
            with open(model_path, 'r') as f:
                self.entity = f.read()
            self.get_logger().info('Loaded entity from: ' + model_path)
        except Exception as e:
            self.get_logger().error('Failed to load entity file: {}'.format(e))
            raise e

    def reset_simulation(self):
        reset_req = Empty.Request()

        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for reset_simulation is not available, waiting ...')

        self.reset_simulation_client.call_async(reset_req)

    def delete_entity(self):
        delete_req = DeleteEntity.Request()
        delete_req.name = self.entity_name

        while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for delete_entity is not available, waiting ...')

        future = self.delete_entity_client.call_async(delete_req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info('A goal deleted.')

    def spawn_entity(self):
        entity_pose = Pose()
        entity_pose.position.x = self.entity_pose_x
        entity_pose.position.y = self.entity_pose_y

        spawn_req = SpawnEntity.Request()
        spawn_req.name = self.entity_name
        spawn_req.xml = self.entity
        spawn_req.initial_pose = entity_pose

        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for spawn_entity is not available, waiting ...')

        future = self.spawn_entity_client.call_async(spawn_req)
        rclpy.spin_until_future_complete(self, future)

    def task_succeed_callback(self, request, response):
        self.delete_entity()
        time.sleep(0.2)
        self.generate_goal_pose()
        time.sleep(0.2)
        self.spawn_entity()
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        self.get_logger().info('A new goal generated.')
        return response

    def task_failed_callback(self, request, response):
        self.delete_entity()
        time.sleep(0.2)
        self.reset_simulation()
        time.sleep(0.2)
        self.generate_goal_pose()
        time.sleep(0.2)
        self.spawn_entity()
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        self.get_logger().info('Environment reset')
        return response

    def initialize_env_callback(self, request, response):
        self.delete_entity()
        time.sleep(0.2)
        self.reset_simulation()
        time.sleep(0.2)
        self.spawn_entity()
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        self.get_logger().info('Environment initialized')
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
    stage = sys.argv[1] if len(sys.argv) > 1 else '1'
    gazebo_interface = GazeboInterface(stage)
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
