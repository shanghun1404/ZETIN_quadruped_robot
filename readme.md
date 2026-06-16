# FOR Quadruped Robot Training

This repository contains the environment configurations and training scripts for a quadruped robot named **"FOR"**, utilizing the IsaacLab and RSL-RL libraries for Deep Reinforcement Learning (RL). 

## 📖 Overview

The goal of this project is to train the "FOR" quadruped robot to navigate various terrains, primarily focusing on flat and rough environments. It uses the **PPO (Proximal Policy Optimization)** algorithm for on-policy reinforcement learning to teach the robot stable locomotion.

### Key Features
- **Dual Environments**: Support for both Flat (`ForFlatEnvCfg`) and Rough (`ForRoughEnvCfg`) terrains.
- **Robust Reward System**: Rewards are carefully crafted for velocity tracking, base height maintenance, and penalizing unsafe joint limits or unnatural sideways gaits.
- **Early Terminations (CaT)**: Curriculum and Termination design instantly terminates episodes if the robot's body, motors, or hips touch the ground, enforcing safer policies.

## ⚙️ Prerequisites

To run these scripts, you need:
- [Isaac Sim / IsaacLab](https://isaac-sim.github.io/IsaacLab/)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)

Make sure you have a working Python virtual environment with Isaac Sim dependencies installed.

## 🚀 Quick Start & Usage

### 1. Training the Robot

To train the robot, simply run the provided shell script or the python training script directly.

```bash
# Run the shell script (Activates venv and runs headless training)
./run_train.sh

# Or run the python script directly (headless mode)
python train.py --headless
```
By default, the training utilizes 2048 parallel environments.

### 2. Validating the Walk (Headless)

To verify the policy by running the latest trained model (`.pt`) and checking if it walks straight to the target:

```bash
python check_walking.py
```

### 3. Visualizing & Recording Video

If you want to evaluate the trained model visually and record the process to an `.mp4` video (optimized for lower-memory setups):

```bash
python play_video.py
```

### 4. Testing Stability

To run a basic physics stability check of the environment:

```bash
python test_stability.py
```

## 🏗️ Project Structure

- `for_env_cfg.py`: Environment definitions (`ForRoughEnvCfg` and `ForFlatEnvCfg`) including observations, rewards, and terminations.
- `for_ppo_cfg.py`: PPO algorithm hyperparameters (Network architecture with GRU, learning rates, etc.).
- `for_robot_cfg.py`: Robot simulation assets loader, actuator setup (`DCMotorCfg`), and joint stiffness/damping configurations.
- `train.py`: Main entry point for training.
- `run_train.sh`: Convenience bash script to activate the environment and start training.
- `check_walking.py`: Script to evaluate forward walking metrics.
- `play_video.py`: Playback and recording utility for the learned policy.
- `test_stability.py`: Simple simulation test.

## 📝 Configuration Details

- **Directionality**: Based on URDF analysis, forward locomotion corresponds to the `-Y` axis, and lateral to the `X` axis.
- **PPO Network**: 
  - Rough Terrain: `[512, 256, 128]` MLP with 128-dim GRU.
  - Flat Terrain: `[128, 128, 128]` MLP with 128-dim GRU.
- **Motor Control**: `stiffness=60.0`, `damping=2.0`, `effort_limit=100.0`.
