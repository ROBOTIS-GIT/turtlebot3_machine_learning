from collections import deque
import random
class HindsightExperienceReplay:
    def __init__(self, maxlen=1000000, batch_size=64):
        self.memory = self.memory = deque(maxlen=maxlen)
        self.n_entrys = 0
        self.batch_size = batch_size

    def append_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        mini_batch = random.sample(self.memory, self.batch_size)