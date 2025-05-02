from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'reward', 'done'])

class Experience_Replay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity) # deque is a double-ended queue, which allows us to add and remove elements from both ends

    def push(self, state, action, next_state, reward, done):
        # check if buffer hit capacity, if yes, remove oldest experience, then add
        # if no, just add 
        self.buffer.append(Transition(state, action, next_state, reward, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        # For compatability with python len() function
        return len(self.buffer)
