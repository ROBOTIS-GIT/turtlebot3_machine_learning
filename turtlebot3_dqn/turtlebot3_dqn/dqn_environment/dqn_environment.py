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
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from turtlebot3_msgs.srv import Dqn
from turtlebot3_msgs.srv import Goal
from std_srvs.srv import Empty
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

import numpy as np
import math


class RLEnvironment(Node):
    """
    A node which has to act as an interface between (rl_agent and gazebo_interface)
    It has to implement make(), step(), and reset() like as an environment implemented in OpenAI gym
    """

    def __init__(self):
        super().__init__('rl_environment')
        """**************************************************************
                                Initialize variables
        **************************************************************"""
        """
            if train_mode is not equal to True it will run on test mode
        """
        self.train_mode = True
        self.goal_pose_x = 0.0
        self.goal_pose_y = 0.0
        self.robot_pose_x = 0.0
        self.robot_pose_y = 0.0

        self.action_size = 5
        # time_out shows the maximum number of actions in each episode
        self.time_out = 1000

        self.done = False
        self.fail = False
        self.succeed = False

        # parameters to calculate the reward
        self.goal_angle = 0.0
        self.goal_distance = 1.0
        self.init_goal_distance = 0.5
        self.scan_ranges = []
        self.min_obstacle_distance = 10.0

        self.local_step = 0
        self.stop_cmd_vel_timer = None
        self.angular_vel = [1.0, 0.5, 0.0, -0.5, -1.0]

        """************************************************************
                 Initialise publisher, subscribers, clients and services
        ************************************************************"""
        qos = QoSProfile(depth=10)

        # Initialize publisher
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)

        # Initialize subscribers
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_sub_callback, qos)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_sub_callback, qos_profile_sensor_data)

        # Initialize client
        self.clients_callback_group = MutuallyExclusiveCallbackGroup()
        self.task_succeed_client = self.create_client(Goal, 'task_succeed', callback_group=self.clients_callback_group)
        self.task_failed_client = self.create_client(Goal, 'task_failed', callback_group=self.clients_callback_group)
        self.initialize_environment_client = self.create_client(Goal, 'initialize_env',
                                                                callback_group=self.clients_callback_group)

        # Initialize service
        self.rl_agent_interface_service = self.create_service(Dqn, 'rl_agent_interface',
                                                              self.rl_agent_interface_callback)

        self.make_environment_service = self.create_service(Empty, 'make_environment', self.make_environment_callback)
        self.reset_environment_service = self.create_service(Dqn, 'reset_environment', self.reset_environment_callback)

    def make_environment_callback(self, request, response):
        """
        gives service to the rl_agent to make the environment by calling initialize_environment() function
        :param request: Empty
        :param response: Empty
        """
        self.initialize_environment()
        return response

    def initialize_environment(self):
        """
        This method will be called just from environment_make_callback()
        initialize_environment_client will send a request to the gazebo_interface service
        the client waits until gets back the response (goal position) form service
        """
        while not self.initialize_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for initialize the environment is not available, waiting ...')

        future = self.initialize_environment_client.call_async(Goal.Request())

        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if not response.success:
            self.get_logger().error('initialize environment request failed')

        else:
            self.goal_pose_x = response.pose_x
            self.goal_pose_y = response.pose_y
            self.get_logger().info('goal initialized at [%f, %f]' % (self.goal_pose_x, self.goal_pose_y))

    def reset_environment_callback(self, request, response):
        """
        gives service to the rl_agent reset environment
        :param request: Dqn request
        :param response: Dqn response (state of the robot: lidar_rays + robot pose (or robots distance to goal and its heading angle to goal))
        :return:
        """
        response.state = self.calculate_state()
        return response

    def call_task_succeed(self):
        """
        When the task is succeed (by reaching the goal) this client will send a request to the gazebo_interface service
        the client waits until gets back the response (goal position) form service
        :return:
        """
        while not self.task_succeed_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for task succeed is not available, waiting ...')

        future = self.task_succeed_client.call_async(Goal.Request())

        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response = future.result()
            self.goal_pose_x = response.pose_x
            self.goal_pose_y = response.pose_y
            self.get_logger().info('service for task succeed finished')
        else:
            self.get_logger().error('task succeed service call failed')

    def call_task_failed(self):
        """
        When the task is failed (either collision or timeout) this client will send a request to the gazebo_interface service
        the client waits until gets back the response (goal position) form service
        :return:
        """
        while not self.task_failed_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for task failed is not available, waiting ...')

        future = self.task_failed_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response = future.result()
            self.goal_pose_x = response.pose_x
            self.goal_pose_y = response.pose_y
            self.get_logger().info('service for task failed finished')
        else:
            self.get_logger().error('task failed service call failed')

    def scan_sub_callback(self, scan):
        """
        subscribes to the laser scanned message
        :param scan: laser scanner message
        :return:
        """
        self.scan_ranges = []  # clear the list
        num_of_lidar_rays = len(scan.ranges)

        for i in range(num_of_lidar_rays):
            if scan.ranges[i] == float('Inf'):
                self.scan_ranges.append(3.5)
            elif np.isnan(scan.ranges[i]):
                self.scan_ranges.append(0)
            else:
                self.scan_ranges.append(scan.ranges[i])

        self.min_obstacle_distance = min(self.scan_ranges)

    def odom_sub_callback(self, msg):
        """
        subscribes to the robots odometry message and calculates the robots distance and heading angle to the goal
        :param msg: robot pose and orientation massage
        :return:
        """
        self.robot_pose_x = msg.pose.pose.position.x
        self.robot_pose_y = msg.pose.pose.position.y
        _, _, self.robot_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        goal_distance = math.sqrt(
            (self.goal_pose_x - self.robot_pose_x) ** 2
            + (self.goal_pose_y - self.robot_pose_y) ** 2)
        path_theta = math.atan2(
            self.goal_pose_y - self.robot_pose_y,
            self.goal_pose_x - self.robot_pose_x)

        goal_angle = path_theta - self.robot_pose_theta
        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi

        elif goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = goal_distance
        self.goal_angle = goal_angle

    def calculate_state(self):
        """
        calculates the robot state (lidar rays , distance to the goal ,robots heading angle toward the goal)
        Checks the task succeed and the task failed
        :return:
        """
        state = list()
        # state.append(float(self.goal_pose_x))
        # state.append(float(self.goal_pose_y))
        state.append(float(self.goal_distance))
        state.append(float(self.goal_angle))

        for var in self.scan_ranges:
            state.append(float(var))
        self.local_step += 1

        # Succeed
        if self.goal_distance < 0.20:  # unit: m
            self.get_logger().info("Goal Reached")
            self.succeed = True
            self.done = True
            self.cmd_vel_pub.publish(Twist())  # robot stop
            self.local_step = 0
            self.call_task_succeed()

        # Fail
        if self.min_obstacle_distance < 0.25:  # unit: m
            self.get_logger().info("Collision happened")
            self.fail = True
            self.done = True
            self.cmd_vel_pub.publish(Twist())  # robot stop
            self.local_step = 0
            self.call_task_failed()

        if self.local_step == self.time_out:
            self.get_logger().info("Time out!")
            self.done = True
            self.local_step = 0
            self.call_task_failed()

        return state

    def calculate_reward(self, action):
        """
        calculates the reward accumulating by agent after doing each action, feel free to change the reward function
        :return:
        """
        if self.train_mode:
            yaw_reward = 1 - 2 * math.sqrt(math.fabs(self.goal_angle / math.pi))

            distance_reward = (2 * self.init_goal_distance) / \
                              (self.init_goal_distance + self.goal_distance) - 1

            obstacle_reward = 0.0
            if self.min_obstacle_distance < 0.50:
                obstacle_reward = -3.0  # self.min_obstacle_distance - 0.45

            # reward = self.action_reward[action] + (0.1 * (2-self.goal_distance)) + obstacle_reward
            reward = distance_reward + obstacle_reward + yaw_reward
            # + for succeed, - for fail
            if self.succeed:
                reward = 15.0
            elif self.fail:
                reward = -10.0
        else:
            if self.succeed:
                reward = 5.0
            elif self.fail:
                reward = -5.0
            else:
                reward = 0.0
        self.get_logger().info('reward: %f ' % reward)
        self.get_logger().info('yaw reward: %f ' % yaw_reward)

        return reward

    def rl_agent_interface_callback(self, request, response):
        """
        gives service to the rl_agent. The rl_agent sends an action as a request and this methods has to does the action
        and gets back the state, reward and done as a response
        :param request: a DQN request including action
        :param response: a DQN response including state, reward, and done
        :return:
        """
        action = request.action
        twist = Twist()
        # robot always receives a (constant linear velocity + a variable angular velocity)
        twist.linear.x = 0.15
        twist.angular.z = self.angular_vel[action]
        self.cmd_vel_pub.publish(twist)
        if self.stop_cmd_vel_timer is None:
            self.stop_cmd_vel_timer = self.create_timer(1.8, self.timer_callback)
        else:
            self.destroy_timer(self.stop_cmd_vel_timer)
            self.stop_cmd_vel_timer = self.create_timer(1.8, self.timer_callback)

        response.state = self.calculate_state()
        response.reward = self.calculate_reward(action)
        response.done = self.done

        if self.done is True:
            self.done = False
            self.succeed = False
            self.fail = False

        return response

    def timer_callback(self):
        """
        after each self.stop_cmd_vel_timer seconds, this method will be called to send a stop cmd_vel to the robot
        :return:
        """
        self.get_logger().info('Stop called')
        self.cmd_vel_pub.publish(Twist())
        self.destroy_timer(self.stop_cmd_vel_timer)

    def euler_from_quaternion(self, quat):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        :param quat: [x, y, z, w]
        :return:
        """

        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)

    rl_environment = RLEnvironment()

    while True:
        rclpy.spin_once(rl_environment)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
