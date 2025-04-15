from collections import namedtuple
import random

transition = namedtuple('transition', ['state', 'action', 'next_state', 'reward'])

class Experience_Replay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, state, action, next_state, reward):
        # check if buffer hit capacity, if yes, remove oldest experience, then add
        # if no, just add 
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition(state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        # For compatability with python len() function
        return len(self.buffer)
