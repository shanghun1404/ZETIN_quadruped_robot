from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

log_dir = "logs/rsl_rl/for_flat"
recent_dir = sorted(os.listdir(log_dir))[-1]
tb_path = os.path.join(log_dir, recent_dir)

ea = EventAccumulator(tb_path)
ea.Reload()
tags = ea.Tags()['scalars']
if "Train/mean_reward" in tags:
    rewards = ea.Scalars("Train/mean_reward")
    lens = ea.Scalars("Train/mean_episode_length")
    print(f"Last 5 Rewards: {[r.value for r in rewards[-5:]]}")
    print(f"Last 5 Ep Lens: {[l.value for l in lens[-5:]]}")
else:
    print("No Train/mean_reward found")
