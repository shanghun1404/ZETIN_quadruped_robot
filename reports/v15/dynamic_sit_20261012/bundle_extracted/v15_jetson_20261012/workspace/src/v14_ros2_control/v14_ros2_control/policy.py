"""50 Hz V14 locomotion policy: frozen V12 depth student + V14 actor, no simulator.

Rebuilds the 235-value IsaacLab policy observation from ROS2 data in the
V8/V11 term order: base linear velocity, base angular velocity, projected
gravity, velocity command, joint position offset, joint velocity, last action
(48), then the V12 depth-predicted height scan (187).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from v12_vision_student import normalize_depth
from v13_depth_policy import compose_depth_policy_observation, load_frozen_v12_student

from v14_ros2_control import contract


ACTION_SCALE = 0.2  # V13LowPassJointPositionActionCfg.scale, default-offset targets
DEPTH_SHAPE = (84, 84)


def load_actor(checkpoint_path: str | Path) -> nn.Sequential:
    """RSL-RL actor MLP (V8 PPO: ELU, no observation normalization) from a checkpoint."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model_state_dict"]
    weights = sorted((int(k.split(".")[1]), k) for k in state if k.startswith("actor.") and k.endswith(".weight"))
    if not weights or any(k.startswith(("actor_obs_normalizer", "std_obs")) for k in state):
        raise ValueError("expected an unnormalized RSL-RL actor MLP")
    layers = []
    for position, (index, key) in enumerate(weights):
        out_features, in_features = state[key].shape
        linear = nn.Linear(in_features, out_features)
        linear.weight.data.copy_(state[key])
        linear.bias.data.copy_(state[f"actor.{index}.bias"])
        layers.append(linear)
        if position < len(weights) - 1:
            layers.append(nn.ELU())
    actor = nn.Sequential(*layers)
    if actor[0].in_features != 235 or actor[-1].out_features != len(contract.JOINT_NAMES):
        raise ValueError("V14 actor must map 235 observations to 12 actions")
    return actor.requires_grad_(False).eval()


def projected_gravity(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """World gravity direction (0, 0, -1) in the base frame (IsaacLab quat_apply_inverse)."""
    w, xyz = quat_wxyz[0], quat_wxyz[1:]
    gravity = quat_wxyz.new_tensor((0.0, 0.0, -1.0))
    t = 2.0 * torch.linalg.cross(xyz, gravity)
    return gravity - w * t + torch.linalg.cross(xyz, t)


class V14DepthPolicy:
    def __init__(self, actor_checkpoint, student_checkpoint, *, device: str = "cpu"):
        self.device = torch.device(device)
        self.actor = load_actor(actor_checkpoint).to(self.device)
        self.student = load_frozen_v12_student(student_checkpoint, str(self.device))
        self.default_pos = torch.tensor([contract.joint_default(n) for n in contract.JOINT_NAMES])
        self.last_action = torch.zeros(len(contract.JOINT_NAMES))

    def reset(self) -> None:
        self.last_action.zero_()

    @torch.inference_mode()
    def step(self, *, joint_pos, joint_vel, lin_vel_b, ang_vel_b, quat_wxyz, command, depth_m):
        """Return (action, joint position target), both in ``contract.JOINT_NAMES`` order."""
        as_tensor = lambda value: torch.as_tensor(value, dtype=torch.float32)
        depth = as_tensor(depth_m)
        if depth.shape != DEPTH_SHAPE:
            raise ValueError(f"depth image must be {DEPTH_SHAPE}, got {tuple(depth.shape)}")
        proprioception = torch.cat((
            as_tensor(lin_vel_b), as_tensor(ang_vel_b), projected_gravity(as_tensor(quat_wxyz)),
            as_tensor(command), as_tensor(joint_pos) - self.default_pos, as_tensor(joint_vel),
            self.last_action,
        ))
        if proprioception.shape != (48,):
            raise ValueError("proprioception must have 48 values")
        pixels = normalize_depth(depth[None, None].to(self.device))
        scan = self.student(pixels)
        teacher = torch.cat((proprioception.to(self.device), torch.zeros(187, device=self.device)))
        observation = compose_depth_policy_observation(teacher[None], scan)
        action = self.actor(observation)[0].cpu()
        if not bool(torch.isfinite(action).all()):
            raise RuntimeError("non-finite policy action")
        self.last_action = action.clone()
        return action, self.default_pos + ACTION_SCALE * action
