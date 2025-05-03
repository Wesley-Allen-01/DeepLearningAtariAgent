from models.dqn import DQN
from experience_replay.experience_replay import Experience_Replay
from experience_replay.experience_replay import Transition
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import yaml
import random
import os
import pickle
class DQNAgent:
    def __init__(self, config):
        # set proper device for training
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")
        
        input_shape = (4, 84, 84)
        
        # iniitalize nets


        lr = config['agent']['learning_rate']
        alpha = config['agent']['alpha']
        eps = config['agent']['eps']
        momentum = config['agent']['momentum']
        
        memory_size = config['agent']['memory_capacity']
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
        self.episodes_so_far = 0
        
        self.policy_net = DQN(input_shape, self.num_actions).to(self.device)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), 
                                       lr=lr, 
                                       alpha=alpha, 
                                       eps=eps, 
                                       momentum=momentum)
        
        
        self.from_checkpoint = self.load_saved_model()
        self.target_net = DQN(input_shape, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
    def get_epsilon(self):
        """ 
        sets epsilon value for epsilon greedy selection
        which is function of steps_done
        """
        slope = (self.epsilon_start - self.epsilon_end) / self.epsilon_frame
        epsilon = self.epsilon_start - slope * self.steps_done
        return max(self.epsilon_end, epsilon)
    
    def select_action(self, state):
        self.epsilon = self.get_epsilon()
        self.steps_done += 1
        # if self.steps_done % 1000 == 0:
        #     print(f"steps done: {self.steps_done}, epsilon: {self.epsilon}")
        # epsilon greedy
        val = random.random()
        if val < self.epsilon:
            action = random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0).div(255)
                q_values = self.policy_net(state)
                action = q_values.argmax(dim=1).item()
                
                
        return action

        
    def optimize_model(self):

        if len(self.memory) < self.batch_size:
            # not enough samples in memory to sample batch
            return
        transitions = self.memory.sample(self.batch_size)
        
        """ 
        so the sample function returns a list of transtions, 
        zip(*transtions) converts the list of namedtuples into a 2d list
        where each val represents a set of states, a set of actions, etc.
        then we use the * operator to unpack the list of tuples into
        separate lists, which we can then use to create a batch of
        transitions. batch is a singluar namedtuple where each field
        (state, action, etc) is a list of the corresponding values
        for each transition in the batch. basically the goal is to have a tensor
        to represent all of the states, a tensor to represent all of the actions, etc.
        """
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.as_tensor(np.array(batch.state), device=self.device, dtype=torch.float32).div(255.0)
        action_batch = torch.as_tensor(np.array(batch.action), device=self.device, dtype=torch.int64).unsqueeze(1)
        next_state_batch = torch.as_tensor(np.array(batch.next_state), device=self.device, dtype=torch.float32).div(255.0)
        reward_batch = torch.as_tensor(np.array(batch.reward), device=self.device, dtype=torch.float32).unsqueeze(1)
        done_batch = torch.as_tensor(np.array(batch.done), device=self.device, dtype=torch.bool).unsqueeze(1)
        
        """
        We say that a q-value of an action is defined recursively as the
        reward received after taking the action plus the max q-value of the
        next state, discounted by a factor gamma. Thus, our loss function is the mse
        between the q-value computed with the policy net and the expected q-value which
        is the reward of the action plus the max q-value of the next state which we calculate 
        with the target net. 
        """
        curr_q_vals = self.policy_net(state_batch).gather(1, action_batch)

        
        with torch.no_grad():
            next_q_vals = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            expected_q_vals = reward_batch + (1 - done_batch.float()) * self.gamma * next_q_vals
            
        loss = F.mse_loss(curr_q_vals, expected_q_vals)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()
        
        """ 
        The reason we have two nets is because we want to have a stable target
        for our q-values. If we only had one net, then the target would be changing
        every time we updated the weights of the net which would cause the q-values
        to oscillate and not converge to the optimal values. We update the policy net every 
        training step, but only update the target net every 10000 steps.
        """
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def checkpoint(self, episodes):
        torch.save({
            'step': episodes,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episodes_so_far': self.episodes_so_far
        }, f"checkpoints/latest.pth")
        with open("checkpoints/replay_buffer.pkl", "wb") as f:
            pickle.dump(self.memory, f)
    
    def load_saved_model(self):
        if not os.path.exists("checkpoints/latest.pth"):
            print("No checkpoint found.")
            return False
        
        checkpoint = torch.load("checkpoints/latest.pth", map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['step']
        self.epsidoes_so_far = checkpoint['episodes_so_far']
        
        self.epsilon = self.get_epsilon()
        
        if not os.path.exists("checkpoints/replay_buffer.pkl"):
            print("No replay buffer found.")
            return False
        else:
            with open("checkpoints/replay_buffer.pkl", "rb") as f:
                self.memory = pickle.load(f)
        print(f"Loaded checkpoint at episode {self.epsidoes_so_far}")
        return True