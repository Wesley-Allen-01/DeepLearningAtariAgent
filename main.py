import gymnasium as gym
import numpy as np
import ale_py
from utils.preprocessing import make_env
from train import train
from agent.dqn_agent import DQNAgent
import yaml
import matplotlib
matplotlib.use("Agg")  # headless: no display on a training server
import matplotlib.pyplot as plt
import pandas as pd


env = make_env("ALE/Breakout-v5")

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

use_wandb = config.get("logging", {}).get("wandb_enabled", False)
if use_wandb:
    import wandb
    wandb.init(project=config["logging"]["wandb_project"], config=config)

agent = DQNAgent(config)
from_checkpoint = agent.from_checkpoint
eps = config["training"]["num_episodes"]
if from_checkpoint:
    print("starting from checkpoint")
    rewards = train(env, agent, episodes=eps, start=agent.episodes_so_far, use_wandb=use_wandb)
else:
    rewards = train(env, agent, episodes=eps, use_wandb=use_wandb)

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
plt.savefig("rewards/rewards_over_time.png")