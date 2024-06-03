#!/usr/bin/env python

# ** Author: khinggan ** 
# ** Email: khinggan2013@gmail.com **

"""
Federated reinforcement learning client local training  
1. Modified ROBOTIS turtlebot3_machine_learning algorithm (https://github.com/ROBOTIS-GIT/turtlebot3_machine_learning) to PyTorch version according to PyTorch Official Tutorial of Reinforcement Learning: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
2. Federated Reinforcement Learning (FRL) Client. 
3. Using ros1_bridge (https://github.com/ros2/ros1_bridge) to separate clients and transmit data using customized service type (LocalTrain)

First, getting request (global model) from the FRL server, then, train it locally, finally upload the trained model to FRL server
"""

import rospy
import numpy as np
import random
import time
from collections import deque, namedtuple
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from turtlebot3_dqn.srv import LocalTrain, LocalTrainResponse
import pickle
import importlib

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

import csv

from script.read_config import yaml_config

# If you want to use CUDA. But, make sure all machines has CUDA compatibility. Otherwise, use cpu
# device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"
print(f"Using {device} device")

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

CURR_CID = 1
STAGE = 1
LOCAL_EPISODES = 2
ROUND = 2

config = yaml_config()        # stages = config['FRL']['server']['stages']

CURR_CID = config['FRL']['client']['curr_cid']
STAGE = config['FRL']['client']['stage']
LOCAL_EPISODES = config['FRL']['client']['local_episode']
ROUND = config['FRL']['server']['round']

stage_module_name = f'src.turtlebot3_dqn.environment_stage_{STAGE}'
# from src.turtlebot3_dqn.environment_stage_1 import Env
Env = getattr(importlib.import_module(stage_module_name), 'Env')

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
        self.state_size = state_size
        self.action_size = action_size

        self.episode_step = 6000
        self.target_update = 300
        self.discount_factor = 0.99
        self.learning_rate = 0.00025
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05

        self.batch_size = 128
        self.train_start = 128
        self.memory = ReplayMemory(10000)

        self.model = DQN(self.state_size, self.action_size).to(device)
        self.target_model = DQN(self.state_size, self.action_size).to(device)
        self.updateTargetModel()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, amsgrad=True)

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

agent = ReinforceAgent(state_size, action_size)

def start_train(request):
    global_model_dict = request.req
    model_dict = pickle.loads(global_model_dict)

    print("#### ROUND {}: CLIENT {} local train on Stage {} #### ".format(request.round, CURR_CID, STAGE))

    # Initialize agent model with global model dict
    agent.model.load_state_dict(model_dict)
    agent.updateTargetModel()
    
    scores, episodes, episode_length, memory_lens, epsilons, episode_hours, episode_minutes, episode_seconds, collisions, goals = [], [], [], [], [], [], [], [], [], []
    global_step = 0
    best_score = 0
    best_model_dict = model_dict

    # start train EPISODES episodes
    start_time = time.time()
    for e in range(1, LOCAL_EPISODES+1):
        done = False
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        score = 0
        collision = 0
        goal = 0
        for t in range(agent.episode_step):
            action = agent.getAction(state)

            next_state, reward, done = env.step(action)

            # check goal or collision
            if reward == 200:
                goal += 1
            
            if reward == -200:
                collision += 1

            reward = torch.tensor([reward], device=device)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            agent.appendMemory(state, action, reward, next_state)

            agent.trainModel()
            score += reward
            state = next_state

            if t >= 180:
                rospy.loginfo("Time out!!")
                done = True

            if done:
                agent.updateTargetModel()
                scores.append(score.item())
                episodes.append(e)
                episode_length.append(t)
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)

                memory_lens.append(len(agent.memory))
                epsilons.append(agent.epsilon)
                episode_hours.append(h)
                episode_minutes.append(m)
                episode_seconds.append(s)
                collisions.append(collision)
                goals.append(goal)

                rospy.loginfo('Ep: %d score: %.2f memory: %d epsilon: %.2f time: %d:%02d:%02d',
                              e, score, len(agent.memory), agent.epsilon, h, m, s)
                # save best model
                if score > best_score:
                    best_score = score
                    best_model_dict = agent.model.state_dict()
                    print("BEST SCORE MODEL SAVE: Episode = {}, Best Score = {}".format(e, best_score))
                break

            global_step += 1
            if global_step % agent.target_update == 0:
                rospy.loginfo("UPDATE TARGET NETWORK")

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
    
    state = env.reset()
    end_time = time.time()

    # SAVE EXPERIMENT DATA
    directory_path = os.environ['ROSFRLPATH'] + "data/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    with open(directory_path + "FRL_localep_{}_totalround_{}_client_{}_stage_{}.csv".format(LOCAL_EPISODES,  ROUND, CURR_CID, STAGE), 'a') as d:
        writer = csv.writer(d)
        writer.writerows([item for item in zip(scores, episodes, memory_lens, epsilons, episode_hours, episode_minutes, episode_seconds, collisions, goals)])
        print([item for item in zip(scores, episodes, memory_lens, epsilons, episode_hours, episode_minutes, episode_seconds, collisions, goals)])

    print("Total Train Time on client {} is : {} seconds".format(CURR_CID, end_time - start_time))
    compressed_model_dict = pickle.dumps(best_model_dict)
    return compressed_model_dict

def handle_local_train(request):
    trained_model_dict = start_train(request)

    response = LocalTrainResponse()

    response.resp = trained_model_dict
    response.cid = CURR_CID
    response.round = request.round
    return response


def client_local_train():
    """client service that get global model, train locally, then return local trained model
    """
    rospy.init_node('client_{}_local_train'.format(CURR_CID))
    s = rospy.Service('client_{}_local_train_service'.format(CURR_CID), LocalTrain, handle_local_train)
    print("Client {} Train global model".format(CURR_CID))
    rospy.spin()

if __name__ == '__main__':
    client_local_train()