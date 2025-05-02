# can just hold our training loop in the form of a function that we can import in our main file
from agent.dqn_agent import DQNAgent
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import time


def train(env, agent, episodes, verbose=False, start=0):
    rewards = []
    print(f"starting at episode {start}")
    try:
        for episode in tqdm(range(start, episodes), initial=start):
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
            
            if (episode+1) % 1000 == 0:
                # print("saving model")
                agent.checkpoint(episode+1)
            
            if verbose and episode % 10 == 0:
                print(f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward}")
            agent.episodes_so_far += 1
    except KeyboardInterrupt:
        agent.checkpoint(episodes)
        print("Training interrupted, model saved.") 
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Saving model before exiting...")
        agent.checkpoint(episodes)
        print("Model saved.")
        raise e
    finally:
        env.close()
        return rewards