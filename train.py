# can just hold our training loop in the form of a function that we can import in our main file
from agent.dqn_agent import DQNAgent



def train(env, agent, episodes):
    rewards = []
    for episode in range(episodes):
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
        print(f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward}")
    env.close()
    return rewards