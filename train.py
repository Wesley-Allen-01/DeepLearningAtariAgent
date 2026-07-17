# can just hold our training loop in the form of a function that we can import in our main file
from datetime import datetime


def train(env, agent, episodes, verbose=False, start=0, use_wandb=False):
    if use_wandb:
        import wandb
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
            agent.episodes_so_far = episode + 1  # count of fully-completed episodes

            if use_wandb:
                metrics = {
                    "episode": episode + 1,
                    "score": total_reward,                      # true (unclipped) game score
                    "score_mean_100": sum(rewards[-100:]) / len(rewards[-100:]),
                    "epsilon": agent.epsilon,
                    "steps_done": agent.steps_done,
                }
                if agent.last_loss is not None:
                    metrics["loss"] = agent.last_loss
                    metrics["avg_q"] = agent.last_q
                wandb.log(metrics, step=agent.steps_done)

            if (episode+1) % 100 == 0:
                # print("saving model")
                with open("training_log.log", "a") as f:
                    f.write(f"[{datetime.now()}] -- Episode {episode +1} completed. \n")
                # snapshot=True lets checkpoint() also drop a numbered ep_{N}.pth
                # snapshot on save_model_freq-episode milestones.
                agent.checkpoint(snapshot=True)

            if verbose and episode % 10 == 0:
                print(f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward}")
    except KeyboardInterrupt:
        agent.checkpoint()
        print("Training interrupted, model saved.")
    except Exception:
        # Save progress, then re-raise so a crash is never silently reported as a
        # successful run (a bare `return` in `finally` would swallow the error).
        agent.checkpoint()
        print("Error during training, model saved. Re-raising.")
        raise
    finally:
        env.close()
    return rewards
