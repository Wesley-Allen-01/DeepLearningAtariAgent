"""
Standalone test (no pytest) for DQNAgent checkpoint save/resume round-trip.

Run with:
    uv run python tests/test_checkpoint_resume.py
"""

import math
import os
import sys
import tempfile

# Insert repo root onto sys.path BEFORE chdir-ing so module imports still work.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402  (must follow the sys.path.insert above)
from agent.dqn_agent import DQNAgent  # noqa: E402

# ---------------------------------------------------------------------------
# Tiny config — keeps memory footprint negligible during tests.
# All required keys from the contract are present.
# ---------------------------------------------------------------------------
SMALL_CONFIG = {
    'agent': {
        'learning_rate': 1e-4,
        'memory_capacity': 256,       # tiny — avoids allocating a real 1M buffer
        'batch_size': 32,
        'replay_start_size': 50,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.1,
        'epsilon_frame': 250000,
        'target_update_freq': 10000,
        'num_actions': 4,
    },
    'training': {
        'update_freq': 4,
        'save_replay_buffer': False,  # don't try to pickle a ~7GB buffer in tests
    },
}

EPSILON_START = SMALL_CONFIG['agent']['epsilon_start']
EPSILON_END   = SMALL_CONFIG['agent']['epsilon_end']
EPSILON_FRAME = SMALL_CONFIG['agent']['epsilon_frame']
STEPS_DONE    = 12345

def expected_epsilon(steps_done):
    """Mirrors the get_epsilon() formula from the contract."""
    slope = (EPSILON_START - EPSILON_END) / EPSILON_FRAME
    return max(EPSILON_END, EPSILON_START - slope * steps_done)


def run_tests():
    original_cwd = os.getcwd()
    tmp_dir = tempfile.mkdtemp()

    try:
        # Work entirely inside a fresh temp dir so we never touch the real checkpoints/.
        os.chdir(tmp_dir)

        # ------------------------------------------------------------------
        # 1. Save a checkpoint
        # ------------------------------------------------------------------
        agent1 = DQNAgent(SMALL_CONFIG)

        # Manually set the fields that checkpoint() should persist.
        agent1.steps_done      = STEPS_DONE
        agent1.episodes_so_far = 7
        agent1.wandb_run_id    = "test-run-abc"

        agent1.checkpoint()  # no arguments — per the contract

        # ------------------------------------------------------------------
        # 2. Verify the atomic write left no .tmp file behind.
        # ------------------------------------------------------------------
        assert not os.path.exists("checkpoints/latest.pth.tmp"), \
            "Temp file checkpoints/latest.pth.tmp was not cleaned up"

        assert os.path.exists("checkpoints/latest.pth"), \
            "checkpoints/latest.pth does not exist after checkpoint()"

        # ------------------------------------------------------------------
        # 3. Load checkpoint via a fresh agent (load_saved_model is called in __init__)
        # ------------------------------------------------------------------
        agent2 = DQNAgent(SMALL_CONFIG)

        # 3a. __init__ should detect and load the checkpoint.
        assert agent2.from_checkpoint is True, \
            f"agent2.from_checkpoint should be True, got {agent2.from_checkpoint!r}"

        # 3b. Scalar counters must be restored exactly.
        assert agent2.steps_done == STEPS_DONE, \
            f"steps_done: expected {STEPS_DONE}, got {agent2.steps_done}"

        assert agent2.episodes_so_far == 7, \
            f"episodes_so_far: expected 7, got {agent2.episodes_so_far}"

        # 3c. W&B run id must survive the round-trip.
        assert agent2.wandb_run_id == "test-run-abc", \
            f"wandb_run_id: expected 'test-run-abc', got {agent2.wandb_run_id!r}"

        # 3d. Epsilon recomputed from restored steps_done — must match formula.
        exp_eps = expected_epsilon(STEPS_DONE)
        assert math.isclose(agent2.get_epsilon(), exp_eps, rel_tol=1e-6), \
            f"get_epsilon(): expected ~{exp_eps:.6f}, got {agent2.get_epsilon():.6f}"

        # 3e. The epsilon stored on agent2 itself (set during __init__) must also match.
        assert math.isclose(agent2.epsilon, exp_eps, rel_tol=1e-6), \
            f"agent2.epsilon: expected ~{exp_eps:.6f}, got {agent2.epsilon:.6f}"

        # 3f. Sanity: resumed epsilon must be well below epsilon_start (exploration
        #     was NOT reset to 1.0 after loading).
        assert agent2.epsilon < EPSILON_START, \
            f"agent2.epsilon {agent2.epsilon:.4f} should be < epsilon_start {EPSILON_START}"

        # 3g. Model weights must be identical — compare first parameter tensor.
        param1 = next(agent1.policy_net.parameters())
        param2 = next(agent2.policy_net.parameters())
        assert torch.equal(param1, param2), \
            "policy_net weights differ between agent1 and agent2 after checkpoint restore"

        print("ALL CHECKPOINT TESTS PASSED")

    finally:
        # Always restore cwd so the caller's environment is undisturbed.
        os.chdir(original_cwd)


if __name__ == "__main__":
    run_tests()
