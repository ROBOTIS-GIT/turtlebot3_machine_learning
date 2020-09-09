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
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from std_srvs.srv import Empty
from turtlebot3_msgs.srv import Goal
from geometry_msgs.msg import Pose

import os
import random
import sys


class GazeboInterface(Node):
    """
        A node which acts as an interface between RL_Environment and Gazebo Simulator

        This nodes mostly receive requests (initializing the environment, task succeed, and task failed) from
        RL_Environment, and gives response to this requests by sending requests to the simulator
    """

    def __init__(self, stage):
        super().__init__('gazebo_interface')
        """**************************************************************
                            Initialize variables
        **************************************************************"""
        # Environment stage (could be 1, 2, 3, 4)
        self.stage = int(stage)

        # Read the 'Goal' Entity Model
        self.entity_name = 'Goal'
        self.entity = None
        self.open_entity()

        # initial entity(Goal) position
        self.entity_pose_x = 0.5
        self.entity_pose_y = 0.0

        """*************************************************************
                Initialize clients and services
        *************************************************************"""

        """
            Initialize clients
            The following clients send their request to Gazebo simulator for
            - deleting entity (Goal)
            - spawn an entity (Goal)
            - reset the simulation environment
        """
        self.delete_entity_client = self.create_client(DeleteEntity, 'delete_entity')
        self.spawn_entity_client = self.create_client(SpawnEntity, 'spawn_entity')
        self.reset_simulation_client = self.create_client(Empty, 'reset_simulation')

        # Initialize services
        """
            Initialize services
            The following services give response to the request of their corresponding client in RLEnvironment class
        """
        self.callback_group = MutuallyExclusiveCallbackGroup()
        self.initialize_env_service = self.create_service(Goal, 'initialize_env', self.initialize_env_callback,
                                                          callback_group=self.callback_group)
        self.task_succeed_service = self.create_service(Goal, 'task_succeed', self.task_succeed_callback,
                                                        callback_group=self.callback_group)
        self.task_failed_service = self.create_service(Goal, 'task_failed', self.task_failed_callback,
                                                       callback_group=self.callback_group)

    def open_entity(self):
        """
        find the path to the goal_box model and loads it
        """
        entity_dir_path = os.path.dirname(os.path.realpath(__file__))
        entity_dir_path = entity_dir_path.replace(
            'turtlebot3_rl/turtlebot3_rl/turtlebot3_gazebo',
            '/turtlebot3/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_dqn_world/goal_box'
        )
        entity_path = os.path.join(entity_dir_path, 'model.sdf')
        self.entity = open(entity_path, 'r').read()

    def reset_simulation(self):
        """
        Sends a request to the gazebo service to reset the simulator
        This method mostly will be called upon a task failed request from RLEnvironment
        """
        reset_req = Empty.Request()

        # check connection to the service server
        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for reset_simulation is not available, waiting ...')

        self.reset_simulation_client.call_async(reset_req)

    def delete_entity(self):
        """
        Deletes the goal_box entity by sending a request to the gazebo service
        This method mostly will be called upon a task success and a task failed request from RLEnvironment
        """
        delete_req = DeleteEntity.Request()
        delete_req.name = self.entity_name

        # check connection to the service server
        while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for delete_entity is not available, waiting ...')

        future = self.delete_entity_client.call_async(delete_req)
        rclpy.spin_until_future_complete(self, future)

    def spawn_entity(self):
        """
        Spawns the goal_box entity by sending a request to the gazebo server
        This method mostly will be called upon a task success and a task failed request from RLEnvironment
        """
        entity_pose = Pose()
        entity_pose.position.x = self.entity_pose_x
        entity_pose.position.y = self.entity_pose_y

        spawn_req = SpawnEntity.Request()
        spawn_req.name = self.entity_name
        spawn_req.xml = self.entity
        spawn_req.initial_pose = entity_pose

        # check connection to the service server
        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for spawn_entity is not available, waiting ...')

        future = self.spawn_entity_client.call_async(spawn_req)
        rclpy.spin_until_future_complete(self, future)

    def task_succeed_callback(self, request, response):
        """
        when a task is succeed (goal is reached), a request will be called from RLEnvironment
        This method has to call some functions to sends back the response (position of the new goal) to the client
        :param request: Empty
        :param response: Position of the new generated goal
        """
        self.delete_entity()
        self.generate_goal_pose()
        self.spawn_entity()
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        self.get_logger().info('A new goal generated.')
        return response

    def task_failed_callback(self, request, response):
        """
        when a task is failed (either a collision happened or timeout reached), a request will be called from RL_environment
        This method has to call some functions to sends back the response (position of the new goal) to the client
        :param request: Empty
        :param response: Position of the new generated goal
        """
        self.delete_entity()
        self.reset_simulation()
        self.generate_goal_pose()
        self.spawn_entity()
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        self.get_logger().info('Environment reset')
        return response

    def initialize_env_callback(self, request, response):
        """
        The RLEnvironment sends a request for initializing the environment
        :param request: Empty
        :param response: Position of the generated goal which will be [0.5, 0]
        """
        self.delete_entity()
        self.reset_simulation()
        self.spawn_entity()
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        self.get_logger().info('Environment initialized')
        return response

    def generate_goal_pose(self):
        """
        generates a random position for the goal if stage 1, 2 and 3
        if stage 4 is called will choose from predefined positions
        """
        if self.stage != 4:
            self.entity_pose_x = random.randrange(-23, 23) / 10
            self.entity_pose_y = random.randrange(-23, 23) / 10
        else:
            goal_pose_list = [[1.0, 0.0], [2.0, -1.5], [0.0, -2.0], [2.0, 2.0], [0.8, 2.0],
                              [-1.9, 1.9], [-1.9, 0.2], [-1.9, -0.5], [-2.0, -2.0], [-0.5, -1.0], [-0.5, 2.0], [2.0, -0.5]]
            rand_index = random.randint(0, 11)
            self.entity_pose_x = goal_pose_list[rand_index][0]
            self.entity_pose_y = goal_pose_list[rand_index][1]


def main(args=sys.argv[1]):
    rclpy.init(args=args)
    gazebo_interface = GazeboInterface(args)
    while True:
        rclpy.spin_once(gazebo_interface, timeout_sec=0.1)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
