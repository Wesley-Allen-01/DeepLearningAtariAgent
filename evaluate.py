# some code to evaluate the model when its done training
import torch
import numy as np

def evaluate(agent, env, num_episodes=100, render=False):
    ### evaluates agent without learning (greedy policy epsilon=0)
    agent.policy_net.eval()
    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            if render:
                env.render()
            
            with torch.no_grad():
                state = torch.tensor(state, device=agent.device).unsqueeze(0).to(agent.device)
                q_values = agent.policy_net(state)
                action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

        rewards.append(episode_reward)
        print(f"Episode {episode + 1}: {episode_reward}")

    agent.policy_net.train()
    return rewards