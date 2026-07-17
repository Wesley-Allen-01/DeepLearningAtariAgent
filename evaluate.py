# some code to evaluate the model when its done training
import random
import torch

def evaluate(agent, env, num_episodes=100, epsilon=0.05):
    ### evaluates agent with a small epsilon-greedy policy (standard DQN eval
    ### protocol). we don't use pure greedy (epsilon=0) here: in breakout the
    ### ball is only relaunched by the FIRE action, and the env only fires
    ### automatically on reset, so a fully greedy policy can get stuck idling
    ### after a life loss until ALE's frame-truncation limit kicks in.
    agent.policy_net.eval()
    rewards = []
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        episode_score = 0
        while not done:
            if random.random() < epsilon:
                action = random.randrange(agent.num_actions)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, device=agent.device, dtype=torch.float32).unsqueeze(0).div(255)
                    q_values = agent.policy_net(state_tensor)
                    action = q_values.argmax(dim=1).item()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            # track the true (unclipped) game score, not the clipped/penalized
            # reward the agent actually trains on
            episode_score += info.get('raw_reward', reward)
            state = next_state

        rewards.append(episode_score)
        print(f"Episode {episode + 1}: {episode_score}")

    agent.policy_net.train()
    return rewards


if __name__ == "__main__":
    from utils.preprocessing import make_env
    from agent.dqn_agent import DQNAgent
    import yaml

    env = make_env("ALE/Breakout-v5")

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # DQNAgent auto-loads checkpoints/latest.pth if one is present
    agent = DQNAgent(config)

    scores = evaluate(agent, env, num_episodes=10)
    print(f"Mean score over {len(scores)} episodes: {sum(scores) / len(scores)}")

    env.close()
