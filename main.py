import gymnasium as gym
import numpy as np
import ale_py
from utils.preprocessing import make_env
from train import train
from agent.dqn_agent import DQNAgent
import yaml
import matplotlib.pyplot as plt
import pandas as pd


env = make_env("ALE/Breakout-v5")

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

agent = DQNAgent(config)

rewards = train(env, agent, episodes=10)
try:
    rewards_so_far = pd.read_csv("rewards/rewards.csv")
except pd.errors.EmptyDataError:
    rewards_so_far = pd.DataFrame(columns=["reward"])

rewards = pd.DataFrame(rewards, columns=["reward"])
rewards = pd.concat([rewards_so_far, rewards], ignore_index=True)

rewards.to_csv("rewards/rewards.csv", index=False)


print("Training Completed")
print(f"Model trained for {agent.steps_done} steps")
print("Rewards saved to rewards/rewards.csv")
plt.plot(rewards)
plt.title("Training Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()