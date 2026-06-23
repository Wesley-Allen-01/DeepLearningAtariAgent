# can just hold our training loop in the form of a function that we can import in our main file
from agent.dqn_agent import DQNAgent
from tqdm import tqdm
import os
import time
from datetime import datetime


def train(env, agent, episodes, verbose=False, start=0):
    rewards = []
    print(f"starting at episode {start}")
    try:
        for episode in range(start, episodes):
            observation, info = env.reset()
            episode_over = False
            total_reward = 0
            while not episode_over:
                action = agent.select_action(observation)
                new_state, reward, terminated, truncated, info, = env.step(action)
                # Log the true game score; the agent still learns from the
                # clipped `reward` stored in the replay buffer below.
                total_reward += info.get('raw_reward', reward)
                episode_over = terminated or truncated
                # Store the NEWEST single frame of the PRE-action observation
                # (FrameStack appends newest last along axis 0, so the newest
                # frame is observation[-1]) together with the action/reward/done
                # of the step taken FROM that observation. The buffer
                # reconstructs the 4-frame state stack (ending at this frame) and
                # next_state stack (ending at the following frame) at sample time.
                agent.memory.push(observation[-1], action, reward, episode_over)
                observation = new_state
                if agent.steps_done % agent.update_freq == 0:
                    agent.optimize_model()
            rewards.append(total_reward)
            
            if (episode+1) % 100 == 0:
                # print("saving model")
                with open("training_log.log", "a") as f:
                    f.write(f"[{datetime.now()}] -- Step {episode +1} completed. \n")
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
