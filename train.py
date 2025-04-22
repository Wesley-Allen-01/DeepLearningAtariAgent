# can just hold our training loop in the form of a function that we can import in our main file
from agent.dqn_agent import DQNAgent
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import time


def train(env, agent, episodes, verbose=False):
    rewards = []
    
    try:
        for episode in tqdm(range(episodes)):
            if episode+1 % 1000 == 0:
                agent.save_model(episode)
            observation, info = env.reset()
            episode_over = False
            total_reward = 0
            while not episode_over:
                action = agent.select_action(observation)
                new_state, reward, terminated, truncated, info, = env.step(action)
                total_reward += reward
                episode_over = terminated or truncated
                agent.memory.push(observation, action, new_state, reward, episode_over)
                observation = new_state
                agent.optimize_model()
            rewards.append(total_reward)

            
            if verbose and episode % 10 == 0:
                print(f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward}")
    except KeyboardInterrupt:
        print("Training interrupted.") 
    finally:
        env.close()
        return rewards