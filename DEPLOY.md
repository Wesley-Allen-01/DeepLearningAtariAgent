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
```bash
git clone https://github.com/Wesley-Allen-01/DeepLearningAtariAgent.git
cd DeepLearningAtariAgent

# The PyTorch base image already has a CUDA torch build — don't let pip replace
# it. Install everything else:
pip install -r requirements.txt

mkdir -p checkpoints rewards
```
If `pip` tries to downgrade torch to a CPU build, install the rest with torch
excluded (e.g. `grep -v '^torch' requirements.txt | pip install -r /dev/stdin`).

## 3. Connect Weights & Biases
```bash
wandb login   # paste your API key when prompted
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
python main.py
# detach with Ctrl-b then d ; reattach later with: tmux attach -t train
```

## 5. Monitor from your phone
Open the **W&B mobile app** (or wandb.ai) and pick the `dqn-breakout` project.
You'll see the reward curve climb live. Random Breakout scores ~1; you should
see `score_mean_100` rise above that within the first ~1–2M steps.

## Checkpointing & resume
- The model/optimizer are saved to `checkpoints/latest.pth` every 100 episodes.
- The ~7 GB replay buffer is **not** saved by default (`save_replay_buffer: false`)
  to keep checkpoints fast and small. On resume the buffer simply re-warms
  (`replay_start_size` steps) before learning continues.
- To resume, just rerun `python main.py` — it auto-loads `latest.pth` and
  continues from the saved episode count.

## Rough cost
At ~$0.30–0.50/hr, a multi-day Breakout run is about **$20–40** total.
