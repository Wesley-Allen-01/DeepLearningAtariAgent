import gymnasium as gym
import numpy as np
import ale_py

# good for initial viz, but need to change render_mode eventually to 
# accelerate training
env = gym.make("ALE/Breakout-v5", render_mode="human")

observation, info = env.reset()

episode_over = False
while not episode_over:
    
    # currently uses random selection to choose action, eventually we 
    # will replace this with a calculation with our agent
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info, = env.step(action)
    episode_over = terminated or truncated
    
env.close() 
    