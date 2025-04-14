from collections import namedtuple
import random

transition = namedtuple('transition', ['state', 'action', 'next_state', 'reward'])

class Experience_Replay:
    def __init__(self, capacity):
        ...
    def push(self, state, action, next_state, reward):
        # check if buffer hit capacity, if yes, remove oldest experience, then add
        # if no, just add 
        ...
    def sample(self, batch_size):
        ...
