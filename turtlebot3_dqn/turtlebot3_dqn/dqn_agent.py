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
import datetime
import json
import math
import os
import random
import sys
import time

import numpy
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty

from turtlebot3_msgs.srv import Dqn


LOGGING = True
current_time = datetime.datetime.now().strftime('[%mm%dd-%H:%M]')
_tensorflow = None
_Dense = None
_Input = None
_MeanSquaredError = None
_load_model = None
_Sequential = None
_Adam = None


def _import_tensorflow():
    try:
        import tensorflow
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import Input
        from tensorflow.keras.losses import MeanSquaredError
        from tensorflow.keras.models import load_model
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
        return (
            tensorflow,
            Dense,
            Input,
            MeanSquaredError,
            load_model,
            Sequential,
            Adam
        )
    except ImportError as e:
        print(f'Error importing TensorFlow: {e}', file=sys.stderr)
        print('Please ensure TensorFlow is properly installed.', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'Fatal error during TensorFlow import: {e}', file=sys.stderr)
        print('This may be due to missing system libraries or incompatible versions.',
              file=sys.stderr)
        sys.exit(1)


def _ensure_tensorflow():
    global _tensorflow, _Dense, _Input, _MeanSquaredError, _load_model, _Sequential, _Adam
    if _tensorflow is None:
        (_tensorflow,
         _Dense,
         _Input,
         _MeanSquaredError,
         _load_model,
         _Sequential,
         _Adam) = _import_tensorflow()


def _create_dqn_metric_class():
    _ensure_tensorflow()
    base_class = _tensorflow.keras.metrics.Metric

    class DQNMetric(base_class):

        def __init__(self, name='dqn_metric'):
            super(DQNMetric, self).__init__(name=name)
            self.loss = self.add_weight(name='loss', initializer='zeros')
            self.episode_step = self.add_weight(name='step', initializer='zeros')

        def update_state(self, y_true, y_pred=0, sample_weight=None):
            self.loss.assign_add(y_true)
            self.episode_step.assign_add(1)

        def result(self):
            return self.loss / self.episode_step

        def reset_states(self):
            self.loss.assign(0)
            self.episode_step.assign(0)

    return DQNMetric


class DQNAgent(Node):

    def __init__(self):
        super().__init__('dqn_agent')
        self.declare_parameter('epsilon_decay', 6000)
        self.declare_parameter('max_training_episodes', 1000)
        self.declare_parameter('model_file', '')
        self.declare_parameter('use_gpu', False)
        self.declare_parameter('verbose', True)
        self.max_training_episodes = self.get_parameter(
            'max_training_episodes'
        ).get_parameter_value().integer_value
        model_file = self.get_parameter('model_file').get_parameter_value().string_value
        use_gpu = self.get_parameter('use_gpu').get_parameter_value().bool_value
        self.verbose = self.get_parameter('verbose').get_parameter_value().bool_value

        DQNMetric = _create_dqn_metric_class()
        _ensure_tensorflow()
        self.tf = _tensorflow
        self.Dense = _Dense
        self.Input = _Input
        self.MeanSquaredError = _MeanSquaredError
        self.load_model = _load_model
        self.Sequential = _Sequential
        self.Adam = _Adam

        if not use_gpu:
            self.tf.config.set_visible_devices([], 'GPU')

        self.train_mode = True
        self.state_size = 26
        self.action_size = 5

        self.done = False
        self.succeed = False
        self.fail = False

        self.discount_factor = 0.99
        self.learning_rate = 0.0007
        self.epsilon = 1.0
        self.step_counter = 0
        self.epsilon_decay = self.get_parameter(
            'epsilon_decay'
        ).get_parameter_value().integer_value
        self.epsilon_min = 0.05
        self.batch_size = 128

        self.replay_memory = collections.deque(maxlen=500000)
        self.min_replay_memory_size = 5000

        self.model = self.create_qnetwork()
        self.use_pretrained_model = bool(model_file)
        self.load_episode = 0
        self.model_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model'
        )
        model_path = os.path.join(
            self.model_dir_path,
            model_file
        )
        if self.use_pretrained_model:
            self.model.set_weights(self.load_model(model_path).get_weights())
            json_path = model_path.replace('.h5', '.json')
            if os.path.exists(json_path):
                with open(json_path) as outfile:
                    param = json.load(outfile)
                    self.epsilon = param.get('epsilon', self.epsilon)
                    self.step_counter = param.get('step_counter', self.step_counter)
                    self.load_episode = param.get('trained_episodes', self.load_episode)
                if self.load_episode >= self.max_training_episodes:
                    self.get_logger().error('Loaded model episode exceeds max training episodes.')
                    raise ValueError('Loaded model episode exceeds max training episodes.')
            else:
                self.get_logger().warn(
                    f'JSON file not found for {model_file}, using default values.'
                )

        self.target_model = self.create_qnetwork()
        self.update_target_after = 5000
        self.target_update_after_counter = 0
        self.update_target_model()

        if LOGGING:
            tensorboard_file_name = current_time + ' dqn_reward'
            home_dir = os.path.expanduser('~')
            dqn_reward_log_dir = os.path.join(
                home_dir, 'turtlebot3_dqn_logs', 'gradient_tape', tensorboard_file_name
            )
            self.dqn_reward_writer = self.tf.summary.create_file_writer(dqn_reward_log_dir)
            self.dqn_reward_metric = DQNMetric()

        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')

        self.action_pub = self.create_publisher(Float32MultiArray, '/get_action', 10)
        self.result_pub = self.create_publisher(Float32MultiArray, 'result', 10)

        self.process()

    def process(self):
        self.env_make()
        time.sleep(1.0)

        episode_num = self.load_episode

        for episode in range(self.load_episode + 1, self.max_training_episodes + 1):
            state = self.reset_environment()
            episode_num += 1
            local_step = 0
            score = 0
            sum_max_q = 0.0

            time.sleep(1.0)

            while True:
                local_step += 1

                q_values = self.model.predict(state, verbose=self.verbose)
                sum_max_q += float(numpy.max(q_values))

                action = int(self.get_action(state))
                next_state, reward, done = self.step(action)
                score += reward

                msg = Float32MultiArray()
                msg.data = [float(action), float(score), float(reward)]
                self.action_pub.publish(msg)

                if self.train_mode:
                    self.append_sample((state, action, reward, next_state, done))
                    self.train_model(done)

                state = next_state

                if done:
                    avg_max_q = sum_max_q / local_step if local_step > 0 else 0.0

                    msg = Float32MultiArray()
                    msg.data = [float(score), float(avg_max_q)]
                    self.result_pub.publish(msg)

                    if LOGGING:
                        self.dqn_reward_metric.update_state(score)
                        with self.dqn_reward_writer.as_default():
                            self.tf.summary.scalar(
                                'dqn_reward', self.dqn_reward_metric.result(), step=episode_num
                            )
                        self.dqn_reward_metric.reset_states()

                    print(
                        'Episode:', episode,
                        'score:', score,
                        'memory length:', len(self.replay_memory),
                        'epsilon:', self.epsilon)

                    param_keys = ['epsilon', 'step_counter', 'trained_episodes']
                    param_values = [self.epsilon, self.step_counter, episode]
                    param_dictionary = dict(zip(param_keys, param_values))
                    break

                time.sleep(0.01)

            if self.train_mode:
                if episode % 100 == 0:
                    idx = 1
                    while True:
                        model_path = os.path.join(
                            self.model_dir_path,
                            f'model{idx}.h5'
                        )
                        json_path = os.path.join(
                            self.model_dir_path,
                            f'model{idx}.json'
                        )
                        if not os.path.exists(model_path):
                            break
                        idx += 1
                    self.model.save(model_path)
                    with open(json_path, 'w') as outfile:
                        json.dump(param_dictionary, outfile)

    def env_make(self):
        while not self.make_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'Environment make client failed to connect to the server, try again ...'
            )

        self.make_environment_client.call_async(Empty.Request())

    def reset_environment(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'Reset environment client failed to connect to the server, try again ...'
            )

        future = self.reset_environment_client.call_async(Dqn.Request())

        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            state = future.result().state
            state = numpy.reshape(numpy.asarray(state), [1, self.state_size])
        else:
            self.get_logger().error(
                'Exception while calling service: {0}'.format(future.exception()))

        return state

    def get_action(self, state):
        if self.train_mode:
            self.step_counter += 1
            self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * math.exp(
                -1.0 * self.step_counter / self.epsilon_decay)
            lucky = random.random()
            if lucky > (1 - self.epsilon):
                result = random.randint(0, self.action_size - 1)
            else:
                result = numpy.argmax(self.model.predict(state, verbose=self.verbose))
        else:
            result = numpy.argmax(self.model.predict(state, verbose=self.verbose))

        return result

    def step(self, action):
        req = Dqn.Request()
        req.action = action

        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('rl_agent interface service not available, waiting again...')

        future = self.rl_agent_interface_client.call_async(req)

        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            next_state = future.result().state
            next_state = numpy.reshape(numpy.asarray(next_state), [1, self.state_size])
            reward = future.result().reward
            done = future.result().done
        else:
            self.get_logger().error(
                'Exception while calling service: {0}'.format(future.exception()))

        return next_state, reward, done

    def create_qnetwork(self):
        model = self.Sequential()
        model.add(self.Input(shape=(self.state_size,)))
        model.add(self.Dense(512, activation='relu'))
        model.add(self.Dense(256, activation='relu'))
        model.add(self.Dense(128, activation='relu'))
        model.add(self.Dense(self.action_size, activation='linear'))
        model.compile(
            loss=self.MeanSquaredError(),
            optimizer=self.Adam(learning_rate=self.learning_rate))
        model.summary()

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_after_counter = 0
        print('*Target model updated*')

    def append_sample(self, transition):
        self.replay_memory.append(transition)

    def train_model(self, terminal):
        if len(self.replay_memory) < self.min_replay_memory_size:
            return
        data_in_mini_batch = random.sample(self.replay_memory, self.batch_size)

        current_states = numpy.array([transition[0] for transition in data_in_mini_batch])
        current_states = current_states.squeeze()
        current_qvalues_list = self.model.predict(current_states, verbose=self.verbose)

        next_states = numpy.array([transition[3] for transition in data_in_mini_batch])
        next_states = next_states.squeeze()
        next_qvalues_list = self.target_model.predict(next_states, verbose=self.verbose)

        x_train = []
        y_train = []

        for index, (current_state, action, reward, _, done) in enumerate(data_in_mini_batch):
            current_q_values = current_qvalues_list[index]

            if not done:
                future_reward = numpy.max(next_qvalues_list[index])
                desired_q = reward + self.discount_factor * future_reward
            else:
                desired_q = reward

            current_q_values[action] = desired_q
            x_train.append(current_state)
            y_train.append(current_q_values)

        x_train = numpy.array(x_train)
        y_train = numpy.array(y_train)
        x_train = numpy.reshape(x_train, [len(data_in_mini_batch), self.state_size])
        y_train = numpy.reshape(y_train, [len(data_in_mini_batch), self.action_size])

        self.model.fit(
            self.tf.convert_to_tensor(x_train, self.tf.float32),
            self.tf.convert_to_tensor(y_train, self.tf.float32),
            batch_size=self.batch_size, verbose=0
        )
        self.target_update_after_counter += 1

        if self.target_update_after_counter > self.update_target_after and terminal:
            self.update_target_model()


def main(args=None):
    rclpy.init(args=args)

    dqn_agent = DQNAgent()
    rclpy.spin(dqn_agent)

    dqn_agent.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
