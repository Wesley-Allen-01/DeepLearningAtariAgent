from models.dqn import DQN
from experience_replay.experience_replay import Experience_Replay
class DQNAgent:
    def __init__(self, config):
        # initalize target and policy nets
        ...
    def select_action(self, state):
        # should handle epsilon greedy here
        ...
    def optimize_model(self):
        # sample batch from experience_replay, eval states using policy, eval next_states using target, 
        # use loss func provided in paper to compute loss, perform grad descent on policy net, update 
        # target net every n steps
        ...
