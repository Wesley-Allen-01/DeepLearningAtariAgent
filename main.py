import gymnasium as gym
import numpy as np
import ale_py
from utils.preprocessing import make_env
from train import train
from agent.dqn_agent import DQNAgent
import yaml

# good for initial viz, but need to change render_mode eventually to 
# accelerate training
env = gym.make("ALE/Breakout-v5", render_mode="human", frameskip=1)

env = make_env("ALE/Breakout-v5")

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

agent = DQNAgent(config)

train(env, agent, episodes=10)


# observation, info = env.reset()

# episode_over = False
# while not episode_over:
    
#     # currently uses random selection to choose action, eventually we 
#     # will replace this with a calculation with our agent
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info, = env.step(action)
#     episode_over = terminated or truncated
    
# env.close() 
    