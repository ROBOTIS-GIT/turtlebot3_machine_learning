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

import collections
import json
import os
import random
import sys
import time

from keras.api.layers import Dense
from keras.api.models import load_model
from keras.api.models import Sequential
from keras.api.optimizers import RMSprop
import numpy
import rclpy
from rclpy.node import Node
import tensorflow

from turtlebot3_msgs.srv import Dqn


class DQNTest(Node):

    def __init__(self, stage):
        super().__init__('dqn_test')

        self.stage = int(stage)
        self.state_size = 26
        self.action_size = 5
        self.episode_size = 3000

        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 64

        self.memory = collections.deque(maxlen=1000000)

        self.model = self.build_model()
        self.target_model = self.build_model()

        self.load_model = True
        self.load_episode = 600
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        self.model_dir_path = os.path.join(base_dir, 'model')
        self.model_path = os.path.join(
            self.model_dir_path,
            'stage'+str(self.stage)+'_episode'+str(self.load_episode)+'.h5')

        if self.load_model:
            loaded_model = load_model(
                self.model_path, custom_objects={'mse': tensorflow.keras.losses.MeanSquaredError()}
            )
            self.model.set_weights(loaded_model.get_weights())
            with open(os.path.join(
                    self.model_dir_path,
                    'stage'+str(self.stage)+'_episode'+str(self.load_episode)+'.json')) as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')

        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')

        self.process()

    def process(self):
        global_step = 0

        for episode in range(self.load_episode+1, self.episode_size):
            global_step += 1
            local_step = 0

            state = list()
            next_state = list()
            done = False
            init = True
            score = 0

            time.sleep(1.0)

            while not done:
                local_step += 1
                if local_step == 1:
                    action = 2
                else:
                    state = next_state
                    action = int(self.get_action(state))

                req = Dqn.Request()
                print(int(action))
                req.action = action
                req.init = init

                while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
                    self.get_logger().info(
                        'rl_agent interface service not available, waiting again...'
                    )
                future = self.rl_agent_interface_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)

                while rclpy.ok():
                    rclpy.spin_once(self)

                    if future.done():

                        if future.result() is not None:
                            # Next state and reward
                            next_state = future.result().state
                            reward = future.result().reward
                            done = future.result().done
                            score += reward
                            init = False
                        else:
                            self.get_logger().error(
                                'Exception while calling service: {0}'.format(future.exception()))

                        break

                time.sleep(0.01)

    def build_model(self):
        model = Sequential()
        model.add(Dense(
            512,
            input_shape=(self.state_size,),
            activation='relu',
            kernel_initializer='lecun_uniform'))
        model.add(Dense(256, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dense(128, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='lecun_uniform'))
        model.compile(
            loss='mse', optimizer=RMSprop(learning_rate=self.learning_rate, rho=0.9, epsilon=1e-06)
        )
        model.summary()

        return model

    def get_action(self, state):
        if numpy.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = numpy.asarray(state)
            q_value = self.model.predict(state.reshape(1, len(state)))
            print(numpy.argmax(q_value[0]))
            return numpy.argmax(q_value[0])

    def train_model(self, target_train_start=False):
        mini_batch = random.sample(self.memory, self.batch_size)
        x_batch = numpy.empty((0, self.state_size), dtype=numpy.float64)
        y_batch = numpy.empty((0, self.action_size), dtype=numpy.float64)

        for i in range(self.batch_size):
            state = numpy.asarray(mini_batch[i][0])
            action = numpy.asarray(mini_batch[i][1])
            reward = numpy.asarray(mini_batch[i][2])
            next_state = numpy.asarray(mini_batch[i][3])
            done = numpy.asarray(mini_batch[i][4])

            q_value = self.model.predict(state.reshape(1, len(state)))
            self.max_q_value = numpy.max(q_value)

            if not target_train_start:
                target_value = self.model.predict(next_state.reshape(1, len(next_state)))
            else:
                target_value = self.target_model.predict(next_state.reshape(1, len(next_state)))

            if done:
                next_q_value = reward
            else:
                next_q_value = reward + self.discount_factor * numpy.amax(target_value)

            x_batch = numpy.append(x_batch, numpy.array([state.copy()]), axis=0)
            y_sample = q_value.copy()
            y_sample[0][action] = next_q_value
            y_batch = numpy.append(y_batch, numpy.array([y_sample[0]]), axis=0)

            if done:
                x_batch = numpy.append(x_batch, numpy.array([next_state.copy()]), axis=0)
                y_batch = numpy.append(y_batch, numpy.array([[reward] * self.action_size]), axis=0)

        self.model.fit(x_batch, y_batch, batch_size=self.batch_size, epochs=1, verbose=0)


def main(args=None):
    if args is None:
        args = sys.argv
    stage = args[1] if len(args) > 1 else '1'
    rclpy.init(args=args)
    dqn_test = DQNTest(stage)
    try:
        while rclpy.ok():
            rclpy.spin_once(dqn_test, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        dqn_test.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
