#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
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

# ** Author: khinggan ** 
# ** Email: khinggan2013@gmail.com **

"""Modification of ROBOTIS turtlebot3_machine_learning algorithm to PyTorch version 
according to PyTorch Official Tutorial of Reinforcement Learning: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import rospy
import os
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque, namedtuple
from std_msgs.msg import Float64
from src.turtlebot3_dqn.environment_stage_1 import Env
# from turtlebot3_dqn.srv import PtModel,PtModelRequest, PtModelResponse
import pickle

import os
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import yaml
import importlib

# Read config.yaml to find the current config.

# Set .pickle directory 

# Global Variables
# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
device = "cpu"
print(f"Using {device} device")

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

# Default values, can be modified by config.yaml
EPISODES = 2
CLIENT_ID = 1
ROUND = 2
STAGE = 1

project_path = "/home/khinggan/my_research/ros_frl"
file_path = project_path + '/config.yaml'
trained_dict_path = project_path + '/ros2_ws/src/frl_server/model_dicts/'

with open(file_path, 'r') as file:
    config = yaml.safe_load(file)
    
config_type = config.get('type')
STAGE = config.get('stage')
    
if config_type == 'FRL':
    frl_config = config.get('frl', {})
    EPISODES = frl_config.get('local_episode')
    ROUND = frl_config.get('round')
    CLIENT_ID = int(frl_config.get('curr_client'))
elif config_type == 'RL':
    rl_config = config.get('rl', {})
    EPISODES = rl_config.get('local_episode')
    ROUND = rl_config.get('round')
    CLIENT_ID = int(rl_config.get('curr_client'))
else:
    print("Invalid type specified in the config file.")

print(f"type: {config_type}, local episode: {EPISODES}, client ID: {CLIENT_ID}")

# stage_module_name = f'src.turtlebot3_dqn.environment_stage_{STAGE}'
# Env = getattr(importlib.import_module(stage_module_name), 'Env')
from src.turtlebot3_dqn.environment_stage_test import Env

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class ReinforceAgent():
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000

        self.model = DQN(self.state_size, self.action_size).to(device)

    def getQvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * np.amax(next_target)

    def getAction(self, state):
        # if np.random.rand() <= 0.1:
        #     return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)
        # else:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            q_value = self.model(state)
        q_value = torch.argmax(q_value).item()
        return q_value
            # q_value = self.model(state.reshape(1, len(state)))
            # self.q_value = q_value
            # return np.argmax(q_value[0])

if __name__ == '__main__':
    rospy.init_node('test_trained_model')

    state_size = 26
    action_size = 5

    env = Env(action_size)

    test_epoches = 10

    agent = ReinforceAgent(state_size, action_size)
    # load trained dict
    model_dict_file_name = "{}_ep_{}_round_{}_stage_{}.pkl".format(config_type, EPISODES, ROUND, STAGE)
    with open(trained_dict_path + model_dict_file_name, 'rb') as md:
        model_dict = pickle.load(md)
    # for key, value in model_dict.items():
    #     print(key, value.size())

    # Initialize agent model with global model dict
    agent.model.load_state_dict(model_dict)
    agent.model.eval()

    scores, episodes = [], []
    global_step = 0
    start_time = time.time()
    collision = 0
    goal = 0

    for e in range(0, test_epoches):
        done = False
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        score = 0
        for t in range(agent.episode_step):
            action = agent.getAction(state)

            next_state, reward, done = env.step(action)

            reward = torch.tensor([reward], device=device)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            score += reward
            state = next_state

            if t >= 500:
                rospy.loginfo("Time out!!")
                done = True

            if done:
                scores.append(score)
                episodes.append(e)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                rospy.loginfo('Ep: %d score: %.2f time: %d:%02d:%02d',
                              e, score, h, m, s)
                
                if reward == 200:
                    goal += 1
                elif reward == -200:
                    collision += 1

                break

            global_step += 1
    
    print("Goal reached = {}, Collision = {}, Goal rate = {} Collision rate = {}".format(goal, collision, goal * 1.0 / test_epoches, collision * 1.0 / test_epoches))