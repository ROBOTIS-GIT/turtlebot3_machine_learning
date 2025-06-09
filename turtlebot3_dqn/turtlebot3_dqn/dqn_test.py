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
# Authors: Ryan Shim, Gilbert, ChanHyeong Lee

import collections
import os
import sys
import time

import numpy
import rclpy
from rclpy.node import Node
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

from turtlebot3_msgs.srv import Dqn


class DQNTest(Node):

    def __init__(self, stage, load_episode):
        super().__init__('dqn_test')

        self.stage = int(stage)
        self.load_episode = int(load_episode)

        self.state_size = 26
        self.action_size = 5

        self.memory = collections.deque(maxlen=1000000)

        self.model = self.build_model()
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model',
            f'stage{self.stage}_episode{self.load_episode}.h5'
        )

        loaded_model = load_model(
            model_path, compile=False, custom_objects={'mse': MeanSquaredError()}
        )
        self.model.set_weights(loaded_model.get_weights())

        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')

        self.run_test()

    def build_model(self):
        model = Sequential()
        model.add(Dense(
            512, input_shape=(self.state_size,),
            activation='relu',
            kernel_initializer='lecun_uniform'
        ))
        model.add(Dense(256, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dense(128, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='lecun_uniform'))
        model.compile(loss=MeanSquaredError(), optimizer=RMSprop(learning_rate=0.00025))
        return model

    def get_action(self, state):
        state = numpy.asarray(state)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return int(numpy.argmax(q_values[0]))

    def run_test(self):
        while True:
            done = False
            init = True
            score = 0
            local_step = 0
            next_state = []

            time.sleep(1.0)

            while not done:
                local_step += 1
                action = 2 if local_step == 1 else self.get_action(next_state)

                req = Dqn.Request()
                req.action = action
                req.init = init

                while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
                    self.get_logger().warn(
                        'rl_agent interface service not available, waiting again...')

                future = self.rl_agent_interface_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)

                if future.done() and future.result() is not None:
                    next_state = future.result().state
                    reward = future.result().reward
                    done = future.result().done
                    score += reward
                    init = False
                else:
                    self.get_logger().error(f'Service call failure: {future.exception()}')

                time.sleep(0.01)


def main(args=None):
    rclpy.init(args=args if args else sys.argv)
    stage = sys.argv[1] if len(sys.argv) > 1 else '1'
    load_episode = sys.argv[2] if len(sys.argv) > 2 else '600'
    node = DQNTest(stage, load_episode)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
