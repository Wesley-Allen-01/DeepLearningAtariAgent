# Deploying a training run (cloud GPU + W&B)

This trains the DQN Breakout agent on a rented GPU and streams live metrics to
Weights & Biases so you can watch progress from the web or the phone app.

## What you need
- A **GPU instance** with **16 GB+ RAM** (the replay buffer alone is ~7 GB).
  A single mid-tier GPU (T4 / A10 / RTX 3060+) is plenty — DQN on one Atari game
  is light on GPU; the limits are RAM and wall-clock time, not FLOPs.
- A **persistent disk** so checkpoints survive a pod restart.
- A free **Weights & Biases** account (https://wandb.ai) and your API key.

## 1. Start the pod (RunPod example)
1. Create a pod from a **PyTorch** template (CUDA + PyTorch preinstalled).
2. Pick a GPU (T4/A10 is fine) and attach a persistent volume mounted at the
   workspace dir.

## 2. Set up the code
This project uses **uv** for dependency management (`pyproject.toml` + `uv.lock`).
```bash
git clone https://github.com/Wesley-Allen-01/DeepLearningAtariAgent.git
cd DeepLearningAtariAgent

# Install uv if the pod doesn't have it:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the venv and install the exact locked dependencies:
uv sync
```
On a Linux GPU pod, `uv` installs the CUDA-enabled `torch` wheel automatically.
Everything below is run through `uv run` so it uses this environment.

## 3. Connect Weights & Biases
```bash
uv run wandb login   # paste your API key when prompted
```
Monitoring is controlled in `config/config.yaml`:
```yaml
logging:
  wandb_enabled: true
  wandb_project: dqn-breakout
```
Set `wandb_enabled: false` to run without it. Logged metrics: true game `score`,
`score_mean_100`, `epsilon`, `loss`, `avg_q`, and `steps_done`.

## 4. Launch detached (survives disconnect)
```bash
tmux new -s train
until uv run python main.py; do
  echo "training crashed at $(date); restarting in 10s"
  sleep 10
done
# detach with Ctrl-b then d ; reattach later with: tmux attach -t train
```

This retry loop ensures that if the training process crashes — due to OOM, a CUDA error, or a network hiccup — the pod restarts it automatically instead of sitting idle and billing. **This is safe:** on each restart, `main.py` auto-loads `checkpoints/latest.pth` and appends to the same W&B run, so a crash costs at most the training progress since the last checkpoint (saved every 100 episodes).

## 5. Monitor from your phone
Open the **W&B mobile app** (or wandb.ai) and pick the `dqn-breakout` project.
You'll see the reward curve climb live. Random Breakout scores ~1; you should
see `score_mean_100` rise above that within the first ~1–2M steps.

To get notified if training crashes, enable alerts in W&B: go to **wandb.ai → User Settings → Notifications**
and turn on "Run crashed" and "Run finished" — these send email and mobile push notifications, so you know immediately if the training stops instead of discovering it hours later.

## Checkpointing & resume
- The model, optimizer, `episodes_so_far`, and `steps_done` are saved to
  `checkpoints/latest.pth` every 100 episodes.  Writes are **atomic** (written
  to a `.tmp` file then renamed via `os.replace`), so a process killed mid-write
  — e.g. a preempted/reclaimed spot instance — can never corrupt the checkpoint.
- Every `save_model_freq` completed episodes (default 1000) a **numbered
  snapshot** `checkpoints/ep_{N}.pth` is written alongside `latest.pth`.  These
  are never overwritten, so if the policy degrades late in training you can roll
  back: copy a good `ep_{N}.pth` over `latest.pth` and resume.  Resume itself
  only ever loads `latest.pth` — snapshots are never auto-loaded.
- `steps_done` is restored on resume, so epsilon-greedy exploration and the
  target-network update schedule continue exactly where they left off.  There is
  no exploration reset between runs.
- The checkpoint stores the **W&B run id**; on resume `main.py` calls
  `wandb.init(..., id=<saved id>, resume="allow")` so the restarted run appends
  to the **same** W&B run/dashboard/charts instead of creating a new fragmented
  run.
- The ~7 GB replay buffer is **not** saved by default (`save_replay_buffer: false`)
  to keep checkpoints fast and small. On resume the buffer simply re-warms
  (`replay_start_size` steps) before learning continues.
- To resume, just rerun `python main.py` (or `uv run python main.py`) — it
  auto-loads `latest.pth` and continues from the saved episode count.

## Rough cost
At ~$0.30–0.50/hr, a multi-day Breakout run is about **$20–40** total.
