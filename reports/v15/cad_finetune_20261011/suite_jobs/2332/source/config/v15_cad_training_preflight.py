"""Strict physical checks, including evidence hidden by finite-rollout cleanup."""
import torch


def validate_preflight_state(nonfinite_steps, data, observations, rewards=None,
                             physics_diverged=None):
    if nonfinite_steps != 0:
        raise RuntimeError(f'Preflight sanitized {nonfinite_steps} nonfinite events')
    if physics_diverged is not None and bool(physics_diverged.any()):
        raise RuntimeError('Preflight encountered physics divergence before episode reset')
    tensors = {name: getattr(data, name) for name in
               ('root_state_w', 'joint_pos', 'joint_vel')}
    tensors.update({f'observation/{name}': value for name, value in observations.items()})
    if rewards is not None:
        tensors['rewards'] = rewards
    for name, value in tensors.items():
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f'Preflight nonfinite {name}')
