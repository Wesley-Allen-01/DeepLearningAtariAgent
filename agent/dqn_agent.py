from models.dqn import DQN
from experience_replay.experience_replay import Experience_Replay
from experience_replay.experience_replay import Transition
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
import random
class DQNAgent:
    def __init__(self, config, env):
        # initalize target and policy nets
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        input_shape = (4, 84, 84)
        self.num_actions = env.action_space.n
        self.policy_net = DQN(input_shape, self.num_actions).to(self.device)
        self.target_net = DQN(input_shape, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        lr = config['agent']['learning_rate']
        alpha = config['agent']['alpha']
        eps = config['agent']['eps']
        momentum = config['agent']['momentum']
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), 
                                       lr=lr, 
                                       alpha=alpha, 
                                       eps=eps, 
                                       momentum=momentum)
        memory_size = config['agent']['replay_capacity']
        self.memory = Experience_Replay(memory_size)
        
        self.steps_done = 0
        self.batch_size = config['agent']['batch_size']
        self.gamma = config['agent']['gamma']
        self.epsilon_start = config['agent']['epsilon_start']
        self.epsilon_end = config['agent']['epsilon_end']
        self.epsilon_frame = config['agent']['epsilon_frame']
        self.target_update_freq = config['agent']['target_update_freq']
        self.epsilon = self.epsilon_start
        self.num_actions = config['agent']['num_actions']
        
        
        
    def get_epsilon(self):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1. * self.steps_done / self.epsilon_frame)
        return epsilon
    
    def select_action(self, state):
        # should handle epsilon greedy here
        self.epsilon = self.get_epsilon()
        self.steps_done += 1
        
        val = random.random()
        if val < self.epsilon:
            action = random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device).unsqueeze(0)
                q_values = self.policy_net(state)
                action = q_values.argmax(dim=1).item()
        
        return action

        
    def optimize_model(self):
        # sample batch from experience_replay, eval states using policy, eval next_states using target, 
        # use loss func provided in paper to compute loss, perform grad descent on policy net, update 
        # target net every n steps
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.tensor(np.array(batch.state), device=self.device)
        action_batch = torch.tensor(np.array(batch.action), device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(np.array(batch.reward), device=self.device).unsqueeze(1)
        next_state_batch = torch.tensor(np.array(batch.next_state), device=self.device)
        
        curr_q_vals = self.policy_net(state_batch).gather(1, action_batch)
        
        with torch.no_grad():
            next_q_vals = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            expected_q_vals = (reward_batch + (self.gamma * next_q_vals))
            
        loss_fn = nn.MSELoss()
        loss = loss_fn(curr_q_vals, expected_q_vals)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
