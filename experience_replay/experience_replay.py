from collections import namedtuple
import random
import numpy as np

# Kept for backwards compatibility with anything still importing Transition.
Transition = namedtuple('Transition', ['state', 'action', 'next_state', 'reward', 'done'])

# Returned by Experience_Replay.sample(); fields are batched numpy arrays.
Batch = namedtuple('Batch', ['state', 'action', 'next_state', 'reward', 'done'])


class Experience_Replay:
    """Frame-based circular replay buffer.

    Instead of storing two full (k, 84, 84) frame stacks per transition (which
    duplicates 3 of 4 frames and explodes memory), we store SINGLE (84, 84)
    uint8 frames once and reconstruct the k-frame stacks at sample time.

    For each stored step i we keep:
        frames[i] : the NEWEST single (84, 84) frame observed AFTER taking the
                    action, i.e. the last frame of the state-stack at step i.
        action[i] : action taken at step i.
        reward[i] : reward received at step i.
        done[i]   : whether the transition starting at frame i was terminal.

    A "state" stack for index i is frames[i-k+1 .. i]. The corresponding
    "next_state" stack is frames[i-k+2 .. i+1]. Stacks are zero-padded at the
    very start of the buffer and across episode boundaries so a stack never
    bleeds across an episode reset.
    """

    def __init__(self, capacity, frame_stack=4):
        self.capacity = capacity
        self.frame_stack = frame_stack

        # Parallel circular structures of fixed length `capacity`.
        self.frames = np.zeros((capacity, 84, 84), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        self.position = 0   # next write index
        self.size = 0       # number of valid stored entries

    def push(self, frame, action, reward, done):
        """Store a single newest (84, 84) uint8 frame plus action/reward/done.

        `frame` must be a single 84x84 frame (NOT a stack). train.py passes
        observation[-1] (the newest frame appended by FrameStack).
        """
        frame = np.asarray(frame, dtype=np.uint8)
        assert frame.shape == (84, 84), f"expected (84,84) frame, got {frame.shape}"

        self.frames[self.position] = frame
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _stack(self, end_index):
        """Build a (k, 84, 84) stack ending at `end_index` (inclusive).

        Frames are gathered going backwards from end_index. We stop extending
        backwards (and zero-pad the rest) if we run off the start of the buffer
        OR cross an episode boundary (a frame whose PRECEDING step was done).
        """
        k = self.frame_stack
        stack = np.zeros((k, 84, 84), dtype=np.uint8)

        # Fill from the newest slot (k-1) back to the oldest (0).
        idx = end_index % self.capacity
        for offset in range(k):
            slot = k - 1 - offset
            stack[slot] = self.frames[idx]

            # Stop if we've gathered enough or there is no earlier frame.
            if offset == k - 1:
                break

            # `dones[idx]` would describe a step starting at idx; but to detect
            # whether the PREVIOUS frame belongs to a prior episode we check the
            # done flag of that previous step. The previous frame is at prev.
            prev = (idx - 1) % self.capacity

            # No earlier real data: we have wrapped past the start of stored data.
            # When the buffer is not full, indices >= size are invalid.
            if self.size < self.capacity:
                if idx == 0:
                    break  # idx-1 would be -1 -> invalid start of buffer
            else:
                # Buffer full: the oldest valid frame sits at self.position.
                if idx == self.position:
                    break

            # Episode boundary: if the previous step was terminal, the frame at
            # `prev` belongs to a prior episode -> stop, zero-pad the rest.
            if self.dones[prev]:
                break

            idx = prev

        return stack

    def _valid_indices(self):
        """Indices i for which a transition (state_i -> next_state_{i+1}) exists.

        We need frame i+1 to be present and to belong to the SAME episode (i.e.
        the i -> i+1 step itself, captured by dones[i], may be terminal — that's
        fine and handled by masking — but i+1 must be a real stored frame).
        """
        if self.size == 0:
            return np.empty(0, dtype=np.int64)

        if self.size < self.capacity:
            # Stored frames occupy indices 0 .. size-1. The newest frame
            # (size-1) has no successor frame, so it is not sampleable.
            return np.arange(0, self.size - 1, dtype=np.int64)
        else:
            # Full buffer: every index has a successor except the newest one,
            # which is at (position - 1). Its successor slot (position) holds the
            # OLDEST frame of a different/older transition, so exclude it.
            newest = (self.position - 1) % self.capacity
            all_idx = np.arange(self.capacity, dtype=np.int64)
            return all_idx[all_idx != newest]

    def sample(self, batch_size):
        valid = self._valid_indices()
        if len(valid) == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        chosen = np.random.choice(valid, size=batch_size, replace=batch_size > len(valid))

        k = self.frame_stack
        state = np.zeros((batch_size, k, 84, 84), dtype=np.uint8)
        next_state = np.zeros((batch_size, k, 84, 84), dtype=np.uint8)
        action = np.zeros(batch_size, dtype=np.int64)
        reward = np.zeros(batch_size, dtype=np.float32)
        done = np.zeros(batch_size, dtype=np.bool_)

        for b, i in enumerate(chosen):
            i = int(i)
            state[b] = self._stack(i)
            action[b] = self.actions[i]
            reward[b] = self.rewards[i]
            done[b] = self.dones[i]
            if self.dones[i]:
                # Terminal transition: next_state is masked by (1-done) in the
                # loss, but must still be a correctly-shaped array. Use zeros.
                next_state[b] = 0
            else:
                next_state[b] = self._stack((i + 1) % self.capacity)

        return Batch(state, action, next_state, reward, done)

    def __len__(self):
        # Number of stored transitions (frames). For compatibility with len().
        return self.size
