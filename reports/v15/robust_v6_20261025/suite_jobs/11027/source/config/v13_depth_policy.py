"""Keep the V12 depth front-end frozen while PPO updates the locomotion actor."""

from __future__ import annotations

from pathlib import Path

import torch

from v12_camera import camera_depth_nchw
from v12_training import checkpoint_is_finite
from v12_vision_student import TinyDepthViT


def compose_depth_policy_observation(
    teacher_observation: torch.Tensor, predicted_scan: torch.Tensor
) -> torch.Tensor:
    if teacher_observation.ndim != 2 or teacher_observation.shape[1] != 235:
        raise ValueError("teacher observation must be [batch, 235]")
    if predicted_scan.shape != (teacher_observation.shape[0], 187):
        raise ValueError("predicted V12 scan must be [batch, 187]")
    return torch.cat((teacher_observation[:, :48], predicted_scan), dim=1)


def load_frozen_v12_student(checkpoint_path: str | Path, device: str) -> TinyDepthViT:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not checkpoint_is_finite(checkpoint):
        raise ValueError("V12 student checkpoint is missing or non-finite")
    model = TinyDepthViT()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False)
    model.eval()
    return model.to(device)


def make_v13_depth_wrapper():
    """Define the IsaacLab wrapper after AppLauncher has started Omniverse."""
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    class V13DepthPolicyWrapper(RslRlVecEnvWrapper):
        def __init__(self, env, *, student_checkpoint: str, clip_actions=None):
            super().__init__(env, clip_actions=clip_actions)
            self._student = load_frozen_v12_student(student_checkpoint, self.device)

        def _with_depth(self, observations):
            with torch.inference_mode():
                predicted_scan = self._student(
                    camera_depth_nchw(self.unwrapped.scene["depth_camera"])
                )
                observations["policy"] = compose_depth_policy_observation(
                    observations["policy"], predicted_scan
                )
            return observations

        def get_observations(self):
            return self._with_depth(super().get_observations())

        def reset(self):
            observations, extras = super().reset()
            return self._with_depth(observations), extras

        def step(self, actions):
            observations, rewards, dones, extras = super().step(actions)
            return self._with_depth(observations), rewards, dones, extras

    return V13DepthPolicyWrapper
