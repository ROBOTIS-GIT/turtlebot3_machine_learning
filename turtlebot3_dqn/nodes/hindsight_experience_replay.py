from collections import deque
import random
import numpy as np


class HindsightExperienceReplay:
    def __init__(self, k=1, strategie="futue", maxlen=1000000, batch_size=64):
        self.k=k
        self.strategie = strategie
        self.episode_replay = []
        self.memory = deque(maxlen=maxlen)
        self.batch_size = batch_size
        self.n_entrys = 0

    def append_episode_replay(self, state, action, goal, position, reward, next_state, done):
        self.episode_replay.append((state, action, goal, position, reward, next_state, done))

    def append_memory(self, transition):
        self.memory.append(transition)
        self.n_entrys += 1

    def sample_memory(self):
        return random.sample(self.memory, self.batch_size)

    def sample_episode_replay(self, t):
        T = len(self.episode_replay)
        if self.strategie == "futute":
            transition_idx = np.random.randint(t, T)
        elif self.strategie == "episode":
            transition_idx = np.random.randint(0, T)
        return self.episode_replay[transition_idx]

    def import_episode(self):
        T = len(self.episode_replay)
        for t in range(T):
            state, action, goal, position, reward, next_state, done = self.episode_replay[t]
            self.append_memory((state, action, goal, reward, next_state, done))
            transitions = self.sample_transitions(t)
            for transition in transitions:
                self.append_memory(transition)
        self.episode_replay=[]

    def sample_transitions(self, t):
        transitions = []
        for _ in range(self.k):
            _, _, _, goal_position, _, _, _ = self.sample_episode_replay(t)
            sample_state, sample_action, _, position, _, sample_next_state, sample_done = self.episode_replay[t]

            # TODO map the following to a reward_function that is passed to the class
            if np.linalg.norm(np.asarray(position) - np.asarray(goal_position)) < 0.13:
                sample_reward = 200
            else:
                sample_reward = 0

            sample_transition = (sample_state, sample_action, goal_position, sample_reward, sample_next_state, sample_done)
            transitions.append(sample_transition)
        return  transitions





            
            
