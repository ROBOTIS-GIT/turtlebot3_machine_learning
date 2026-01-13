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
# Authors: Ryan Shim, Gilbert, ChanHyeong Lee, Hyungyu Kim

import collections
import os
import sys
import time

import numpy
import rclpy
from rclpy.node import Node

from turtlebot3_msgs.srv import Dqn

_tensorflow = None
_Dense = None
_MeanSquaredError = None
_load_model = None
_Sequential = None
_RMSprop = None


def _import_tensorflow():
    """Lazy import TensorFlow and Keras modules."""
    try:
        import tensorflow
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.losses import MeanSquaredError
        from tensorflow.keras.models import load_model
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import RMSprop
        return (
            tensorflow,
            Dense,
            MeanSquaredError,
            load_model,
            Sequential,
            RMSprop
        )
    except ImportError as e:
        print(f'Error importing TensorFlow: {e}', file=sys.stderr)
        print('Please ensure TensorFlow is properly installed.', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'Fatal error during TensorFlow import: {e}', file=sys.stderr)
        print(
            'This may be due to missing system libraries or incompatible versions.',
            file=sys.stderr)
        sys.exit(1)


def _ensure_tensorflow():
    """Ensure TensorFlow is imported and stored in global variables."""
    global _tensorflow, _Dense, _MeanSquaredError, _load_model, _Sequential, _RMSprop
    if _tensorflow is None:
        (_tensorflow,
         _Dense,
         _MeanSquaredError,
         _load_model,
         _Sequential,
         _RMSprop) = _import_tensorflow()


class DQNTest(Node):

    def __init__(self):
        super().__init__('dqn_test')
        self.declare_parameter('model_file', '')
        self.declare_parameter('use_gpu', False)
        self.declare_parameter('verbose', True)
        model_file = self.get_parameter('model_file').get_parameter_value().string_value
        use_gpu = self.get_parameter('use_gpu').get_parameter_value().bool_value
        self.verbose = self.get_parameter('verbose').get_parameter_value().bool_value

        # Lazy import TensorFlow and store as instance variables
        _ensure_tensorflow()
        self.tf = _tensorflow
        self.Dense = _Dense
        self.MeanSquaredError = _MeanSquaredError
        self.load_model = _load_model
        self.Sequential = _Sequential
        self.RMSprop = _RMSprop

        if not use_gpu:
            self.tf.config.set_visible_devices([], 'GPU')

        self.state_size = 26
        self.action_size = 5

        self.memory = collections.deque(maxlen=1000000)

        self.model = self.build_model()
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model',
            model_file
        )

        loaded_model = self.load_model(
            model_path, compile=False, custom_objects={'mse': self.MeanSquaredError()}
        )
        self.model.set_weights(loaded_model.get_weights())

        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')

        self.run_test()

    def build_model(self):
        model = self.Sequential()
        model.add(self.Dense(
            512, input_shape=(self.state_size,),
            activation='relu',
            kernel_initializer='lecun_uniform'
        ))
        model.add(self.Dense(256, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(self.Dense(128, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(
            self.Dense(
                self.action_size,
                activation='linear',
                kernel_initializer='lecun_uniform'))
        model.compile(loss=self.MeanSquaredError(), optimizer=self.RMSprop(learning_rate=0.00025))
        return model

    def get_action(self, state):
        state = numpy.asarray(state)
        q_values = self.model.predict(state.reshape(1, -1), verbose=self.verbose)
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
    node = DQNTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
