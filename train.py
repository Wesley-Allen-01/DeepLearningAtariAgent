# can just hold our training loop in the form of a function that we can import in our main file
from agent.dqn_agent import DQNAgent

def train(config, env, agent, episodes, device):
    episode_over = False
    for episode in range(episodes):
        observation, info = env.reset()
        while not episode_over:
            action = agent.select_action(observation)
            new_state, reward, terminated, truncated, info, = env.step(action)
            agent.memory.push(observation, action, new_state, reward)
            observation = new_state
            agent.optimize_model()
            episode_over = terminated or truncated
        env.close()
    