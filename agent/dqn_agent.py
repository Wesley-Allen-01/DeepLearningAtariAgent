from models.dqn import DQN
from experience_replay.experience_replay import Experience_Replay
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import pickle
import shutil
class DQNAgent:
    def __init__(self, config):
        # set proper device for training
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")
        
        input_shape = (4, 84, 84)
        
        # iniitalize nets


        lr = config['agent']['learning_rate']

        memory_size = config['agent']['memory_capacity']
        self.memory = Experience_Replay(memory_size)
        
        self.steps_done = 0
        self.batch_size = config['agent']['batch_size']
        self.replay_start_size = config['agent']['replay_start_size']
        self.gamma = config['agent']['gamma']
        self.epsilon_start = config['agent']['epsilon_start']
        self.epsilon_end = config['agent']['epsilon_end']
        self.epsilon_frame = config['agent']['epsilon_frame']
        self.target_update_freq = config['agent']['target_update_freq']
        self.update_freq = config['training']['update_freq']
        self.epsilon = self.epsilon_start
        self.num_actions = config['agent']['num_actions']
        self.episodes_so_far = 0
        self.save_replay_buffer = config['training'].get('save_replay_buffer', False)
        # Keep a numbered checkpoint snapshot every this many completed episodes
        # (in addition to latest.pth) so a degraded run can be rolled back. <=0
        # disables snapshots.
        self.save_model_freq = config['training'].get('save_model_freq', 1000)

        # Most-recent training stats, surfaced for logging (None until first update).
        self.last_loss = None
        self.last_q = None

        # W&B run id, persisted in checkpoints so a resume continues the same run.
        self.wandb_run_id = None
        
        self.policy_net = DQN(input_shape, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        
        self.from_checkpoint = self.load_saved_model()
        # Milestone index of the most recent numbered snapshot. Seeded from the
        # (possibly restored) episode count so a resume doesn't re-snapshot
        # milestones already passed. See checkpoint().
        if self.save_model_freq > 0:
            self._last_snapshot_milestone = self.episodes_so_far // self.save_model_freq
        else:
            self._last_snapshot_milestone = 0
        self.target_net = DQN(input_shape, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
    def get_epsilon(self):
        """ 
        sets epsilon value for epsilon greedy selection
        which is function of steps_done
        """
        slope = (self.epsilon_start - self.epsilon_end) / self.epsilon_frame
        epsilon = self.epsilon_start - slope * self.steps_done
        return max(self.epsilon_end, epsilon)
    
    def select_action(self, state):
        self.epsilon = self.get_epsilon()
        self.steps_done += 1
        # if self.steps_done % 1000 == 0:
        #     print(f"steps done: {self.steps_done}, epsilon: {self.epsilon}")
        # epsilon greedy
        val = random.random()
        if val < self.epsilon:
            action = random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0).div(255)
                q_values = self.policy_net(state)
                action = q_values.argmax(dim=1).item()
                
                
        return action

        
    def optimize_model(self):

        if len(self.memory) < self.replay_start_size:
            # warm-up: collect enough random experience before any gradient updates
            return
        batch = self.memory.sample(self.batch_size)

        """
        The frame-based buffer's sample() returns a Batch of already-batched
        numpy arrays (state/next_state are (B, 4, 84, 84) uint8 stacks
        reconstructed on the fly; action/reward/done are (B,)). We just convert
        them to float tensors on-device, dividing the image stacks by 255.
        """
        state_batch = torch.as_tensor(batch.state, device=self.device, dtype=torch.float32).div(255.0)
        action_batch = torch.as_tensor(batch.action, device=self.device, dtype=torch.int64).unsqueeze(1)
        next_state_batch = torch.as_tensor(batch.next_state, device=self.device, dtype=torch.float32).div(255.0)
        reward_batch = torch.as_tensor(batch.reward, device=self.device, dtype=torch.float32).unsqueeze(1)
        done_batch = torch.as_tensor(batch.done, device=self.device, dtype=torch.bool).unsqueeze(1)
        
        """
        We say that a q-value of an action is defined recursively as the
        reward received after taking the action plus the max q-value of the
        next state, discounted by a factor gamma. Thus, our loss function is the mse
        between the q-value computed with the policy net and the expected q-value which
        is the reward of the action plus the max q-value of the next state which we calculate 
        with the target net. 
        """
        curr_q_vals = self.policy_net(state_batch).gather(1, action_batch)

        
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(1, keepdim=True)
            next_q_vals = self.target_net(next_state_batch).gather(1, next_actions)
            expected_q_vals = reward_batch + (1 - done_batch.float()) * self.gamma * next_q_vals
            
        loss = F.smooth_l1_loss(curr_q_vals, expected_q_vals)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        # Stash for monitoring (e.g. Weights & Biases) without affecting training.
        self.last_loss = loss.item()
        self.last_q = curr_q_vals.mean().item()
        
        """ 
        The reason we have two nets is because we want to have a stable target
        for our q-values. If we only had one net, then the target would be changing
        every time we updated the weights of the net which would cause the q-values
        to oscillate and not converge to the optimal values. We update the policy net every 
        training step, but only update the target net every 10000 steps.
        """
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def checkpoint(self, snapshot=False):
        os.makedirs("checkpoints", exist_ok=True)
        # Write to a temp file then atomically rename, so a process killed
        # mid-write (e.g. a reclaimed spot instance) can't corrupt latest.pth.
        tmp_path = "checkpoints/latest.pth.tmp"
        torch.save({
            'steps_done': self.steps_done,
            'episodes_so_far': self.episodes_so_far,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'wandb_run_id': self.wandb_run_id,
        }, tmp_path)
        os.replace(tmp_path, "checkpoints/latest.pth")
        # latest.pth is overwritten forever, but DQN training isn't monotonic:
        # if the policy degrades late there's otherwise no way back. So every
        # save_model_freq completed episodes, keep a numbered snapshot next to
        # latest.pth (never overwritten). We copy the checkpoint we just wrote,
        # so the snapshot is byte-identical to latest.pth and loads the same way.
        # Only the regular per-episode checkpoint requests snapshots; the
        # crash/interrupt paths just refresh latest.pth.
        if snapshot and self.save_model_freq > 0:
            milestone = self.episodes_so_far // self.save_model_freq
            if milestone > self._last_snapshot_milestone:
                shutil.copy2("checkpoints/latest.pth",
                             f"checkpoints/ep_{self.episodes_so_far}.pth")
                self._last_snapshot_milestone = milestone
        # The replay buffer is ~7GB; only persist it when explicitly enabled.
        # Without it, a resumed run simply re-warms the buffer before learning.
        if self.save_replay_buffer:
            with open("checkpoints/replay_buffer.pkl", "wb") as f:
                pickle.dump(self.memory, f)
    
    def load_saved_model(self):
        if not os.path.exists("checkpoints/latest.pth"):
            print("No checkpoint found.")
            return False
        
        checkpoint = torch.load("checkpoints/latest.pth", map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.episodes_so_far = checkpoint['episodes_so_far']
        self.wandb_run_id = checkpoint.get('wandb_run_id')

        self.epsilon = self.get_epsilon()

        if os.path.exists("checkpoints/replay_buffer.pkl"):
            with open("checkpoints/replay_buffer.pkl", "rb") as f:
                self.memory = pickle.load(f)
        else:
            print("No replay buffer found; will re-warm before learning.")
        print(f"Loaded checkpoint at episode {self.episodes_so_far}")
        return True