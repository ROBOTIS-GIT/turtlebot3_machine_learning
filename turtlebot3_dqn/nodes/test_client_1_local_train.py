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
import json
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque, namedtuple
from std_msgs.msg import Float64
from src.turtlebot3_dqn.environment_stage_1 import Env
# from turtlebot3_dqn.srv import PtModel,PtModelRequest, PtModelResponse
from turtlebot3_dqn.srv import S2CPtModel, S2CPtModelRequest, S2CPtModelResponse
from turtlebot3_dqn.srv import C2SPtModel, C2SPtModelRequest, C2SPtModelResponse
from turtlebot3_dqn.srv import LocalTrain, LocalTrainRequest, LocalTrainResponse
import pickle

import os
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


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

# need to change EPISODES, CLIENT_ID
EPISODES = 100
CLIENT_ID = 1
state_size = 26
action_size = 5
env = Env(action_size)

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
        self.load_model = False
        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 300
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 64
        self.memory = ReplayMemory(10000)

        self.model = DQN(self.state_size, self.action_size).to(device)
        self.target_model = DQN(self.state_size, self.action_size).to(device)
        self.updateTargetModel()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, amsgrad=True)

        # if self.load_model:
        #     self.model.set_weights(load_model(self.dirPath+str(self.load_episode)+".h5").get_weights())

        #     with open(self.dirPath+str(self.load_episode)+'.json') as outfile:
        #         param = json.load(outfile)
        #         self.epsilon = param.get('epsilon')

    def getQvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.discount_factor * np.amax(next_target)

    def updateTargetModel(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def getAction(self, state):
        if np.random.rand() <= self.epsilon:
            return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                self.q_value = self.model(state).max(1).indices.view(1, 1)
                return self.q_value
            # q_value = self.model(state.reshape(1, len(state)))
            # self.q_value = q_value
            # return np.argmax(q_value[0])

    def appendMemory(self, state, action, reward, next_state):
        self.memory.push(state, action, reward, next_state)

    def trainModel(self):
        if self.memory.__len__() < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), device=device, dtype=torch.bool)
        # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        # next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values = self.target_model(next_state_batch).max(1).values
        # with torch.no_grad():
        #     next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()

pub_result = rospy.Publisher('/result', Float64, queue_size=5)
result = Float64()
agent = ReinforceAgent(state_size, action_size)

def start_train(request):
    global_model_dict = request.req
    model_dict = pickle.loads(global_model_dict)
    # for key, value in model_dict.items():
    #     print(key, value.size())

    print("Client {} Round {}".format(CLIENT_ID, request.round))

    # Initialize agent model with global model dict
    agent.model.load_state_dict(model_dict)
    agent.updateTargetModel()
    
    scores, episodes = [], []
    global_step = 0

    # start train EPISODES episodes
    start_time = time.time()
    for e in range(agent.load_episode, EPISODES):
        done = False
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        score = 0
        for t in range(agent.episode_step):
            action = agent.getAction(state)

            next_state, reward, done = env.step(action)

            reward = torch.tensor([reward], device=device)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            agent.appendMemory(state, action, reward, next_state)

            agent.trainModel()
            score += reward
            state = next_state

            if t >= 500:
                rospy.loginfo("Time out!!")
                done = True

            if done:
                result.data = score  # original version: result.data = [score, np.max(agent.q_value)]
                pub_result.publish(result)
                agent.updateTargetModel()
                scores.append(score)
                episodes.append(e)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
                              e, score, len(agent.memory), agent.epsilon, h, m, s)
                break

            global_step += 1
            if global_step % agent.target_update == 0:
                rospy.loginfo("UPDATE TARGET NETWORK")

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
    
    state = env.reset()
    end_time = time.time()

    print("Total Train Time on client {} is : {} seconds".format(CLIENT_ID, end_time - start_time))
    compressed_model_dict = pickle.dumps(agent.model.state_dict())
    return compressed_model_dict

def handle_local_train(request):
    trained_model_dict = start_train(request)

    response = LocalTrainResponse()

    response.resp = trained_model_dict
    response.cid = CLIENT_ID
    response.round = request.round
    return response


def client_local_train():
    rospy.init_node('client_{}_local_train'.format(CLIENT_ID))

    s = rospy.Service('client_{}_local_train_service'.format(CLIENT_ID), LocalTrain, handle_local_train)
    print("Client {} Train global model".format(CLIENT_ID))
    rospy.spin()

if __name__ == '__main__':
    client_local_train()