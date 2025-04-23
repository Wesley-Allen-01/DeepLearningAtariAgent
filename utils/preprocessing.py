import gymnasium as gym
from gymnasium import ObservationWrapper
import numpy as np
import ale_py
from gymnasium.spaces import Box
import cv2
from collections import deque

# logic for preprocessing frames using methods outlined in paper
# im pretty sure they convert each frame to gray scale, and reduce dimensions to 84x84, and then frame stacking

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        # creates np array that can store 2 unprocessed frames
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
    
    def step(self, action):
        total_reward = 0.0
        terminated, truncated = False, False
        
        for i in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            if i >= (self._skip - 2):
                idx = i - (self._skip - 2)
                self._obs_buffer[idx] = obs
            total_reward += reward
            
            if term or trunc:
                terminated = term
                truncated = trunc
                break
        max_frame = np.maximum(self._obs_buffer[0], self._obs_buffer[1])
        return max_frame, total_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer[:] = 0
        return obs, info


class PreprocessFrame(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(84,84), dtype=np.uint8)
    def observation(self, observation):
        obs = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self._k = k
        self.frames = deque(maxlen=k)
        shape = self.env.observation_space.shape
        self.observation_space = Box(low=0, high=255, shape=(k, shape[0], shape[1]), dtype=np.uint8)
        
    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self._k):
            self.frames.append(ob)
        return self._get_obs(), info
    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_obs(), reward, terminated, truncated, info
    def _get_obs(self):
        return np.stack(self.frames, axis=0)
        
class LifeLossPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, life_loss_penalty=1.0):
        super().__init__(env)
        self._life_loss_penalty = life_loss_penalty
        self.lives = 0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.lives = info.get('lives', 0)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_lives = info.get('lives', 0)
        
        if new_lives < self.lives:
            reward -= self._life_loss_penalty
        
        self.lives = new_lives
            
        return obs, reward, terminated, truncated, info

def make_env(env_name, frame_stack=4):
    env = gym.make(env_name, frameskip=1)
    env = MaxAndSkipEnv(env)
    env = PreprocessFrame(env)
    env = LifeLossPenaltyWrapper(env)
    env = FrameStack(env, frame_stack)
    return env


if __name__ == "__main__":
    env = make_env("ALE/Breakout-v5", frame_stack=4)
    observation, info = env.reset()
    assert observation.shape == (4, 84, 84), "Observation shape is not as expected."
    env.close()