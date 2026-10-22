"""Frozen-base residual adapter using the existing RSL VecEnv interface."""

import hashlib
import json
import math
from pathlib import Path

import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

from v14_residual_math import bounded_delta, capsule_gaps, compose_action


PROFILE_NAMES = tuple(f"{leg}_{part}_profile" for leg in ("FL", "FR", "RL", "RR") for part in ("thigh", "calf"))


def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def module_digest(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def initialize_residual_actor(policy):
    last = [m for m in policy.actor.modules() if isinstance(m, torch.nn.Linear)][-1]
    torch.nn.init.zeros_(last.weight)
    torch.nn.init.zeros_(last.bias)


def save_initial_residual(runner, path):
    initialize_residual_actor(runner.alg.policy)
    # Installed RSL-RL save() needs the logging state normally created by learn().
    runner._prepare_logging_writer()
    runner.save(str(path))


def resume_residual(runner, checkpoint, output_checkpoint):
    """Continue a validated trained checkpoint without re-zeroing its actor."""
    runner.load(str(checkpoint), load_optimizer=True, map_location=runner.device)
    completed = runner.current_learning_iteration
    if not isinstance(completed, int) or completed < 0 or not runner.alg.optimizer.state:
        raise ValueError('resume requires a completed training update and optimizer state')
    if not all(bool(torch.isfinite(p).all()) for p in runner.alg.policy.parameters()):
        raise ValueError('non-finite resumed policy')
    runner._prepare_logging_writer()
    runner.save(str(output_checkpoint))
    # Installed RSL-RL saves the index of the update it has already completed.
    runner.current_learning_iteration = completed + 1
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=file_digest(checkpoint),
                loaded_iteration=completed, next_iteration=completed+1,
                policy_tensor_sha256=module_digest(runner.alg.policy), optimizer_restored=True)


def seal_residual_checkpoints(directory):
    directory = Path(directory)
    sidecar = directory/'v14_residual_manifest.json'
    data = json.loads(sidecar.read_text())
    data['checkpoints'] = {p.name:file_digest(p) for p in sorted(directory.glob('model_*.pt'))}
    temporary = sidecar.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(sidecar)


class ResidualOnPolicyRunner(OnPolicyRunner):
    """Publish complete checkpoints with the binding needed by live evaluations."""

    def save(self, path, infos=None):
        path = Path(path)
        temporary = path.with_suffix('.pt.tmp')
        super().save(str(temporary), infos)
        temporary.replace(path)
        seal_residual_checkpoints(path.parent)


def residual_agent_cfg(base_cfg):
    import copy

    cfg = copy.deepcopy(base_cfg)
    cfg.experiment_name = 'guide_v14_residual'
    cfg.run_name = 'frozen5700_gap_residual'
    cfg.save_interval = 50
    cfg.policy.actor_hidden_dims = [128, 128]
    cfg.policy.critic_hidden_dims = [128, 128]
    cfg.policy.init_noise_std = .05
    cfg.policy.noise_std_type = 'log'
    cfg.policy.actor_obs_normalization = False
    cfg.policy.critic_obs_normalization = False
    cfg.algorithm.learning_rate = 1e-4
    cfg.algorithm.schedule = 'fixed'
    cfg.algorithm.entropy_coef = 0.
    return cfg


def load_residual_manifest(checkpoint, *, base_checkpoint, student_checkpoint, usd_path, heading_control):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    data = json.loads((checkpoint.parent/'v14_residual_manifest.json').read_text())
    if (data.get('schema') != 1 or data.get('observation_dim') != 307
            or data.get('actor_hidden_dims') != [128,128] or data.get('action_scale') != .2
            or data.get('heading_control') != heading_control):
        raise ValueError('incompatible residual observation/action/heading configuration')
    bound = data.get('max_rad', float('nan'))
    if not isinstance(bound, (float,int)) or not math.isfinite(bound) or bound != .05:
        raise ValueError('schema1 requires reviewed .05 rad residual bound')
    if data.get('checkpoints', {}).get(checkpoint.name) != file_digest(checkpoint):
        raise ValueError('residual checkpoint digest is missing or mismatched')
    for key, path in (('base_sha256',base_checkpoint), ('student_sha256',student_checkpoint), ('usd_sha256',usd_path)):
        if data.get(key) != file_digest(path):
            raise ValueError(f'residual frozen input mismatch: {key}')
    return data


def robot_capsule_gaps(env):
    robot = env.scene['robot']
    if not hasattr(env, '_v14_residual_profile_ids'):
        env._v14_residual_profile_ids = [robot.body_names.index(name) for name in PROFILE_NAMES]
    ids = env._v14_residual_profile_ids
    return capsule_gaps(robot.data.body_pos_w[:, ids], robot.data.body_quat_w[:, ids])


class ResidualPolicyWrapper:
    """PPO stores raw residual samples; the simulator and base observe combined actions."""

    def __init__(self, env, frozen_actor, *, max_rad=.05):
        self.env = env
        self.frozen_actor = frozen_actor.requires_grad_(False).eval()
        self.max_rad = max_rad
        self.action_scale = float(env.unwrapped.cfg.actions.joint_pos.scale)
        if self.action_scale != .2:
            raise ValueError("residual policy requires reviewed action scale .2")
        self.previous_delta = torch.zeros(env.num_envs, 12, device=env.device)
        bounded_delta(self.previous_delta, max_rad)  # Validate even for zero-output evaluation.
        self.previous_gaps = None
        self.base_action = None
        self.cached_observation = None
        env.unwrapped._v14_residual_delta = self.previous_delta.clone()
        env.unwrapped._v14_residual_previous = self.previous_delta.clone()
        env.unwrapped._v14_residual_max_rad = max_rad
        self.max_observed_delta_rad = 0.

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    @torch.no_grad()
    def _build_observation(self, obs, dones=None):
        original = obs['policy']
        if original.shape != (self.num_envs, 235):
            raise ValueError("frozen base requires 235-dimensional depth observations")
        self.base_action = self.frozen_actor.act_inference(obs).detach()
        gaps = robot_capsule_gaps(self.unwrapped)
        closing = (self.previous_gaps-gaps)/self.unwrapped.step_dt if self.previous_gaps is not None else torch.zeros_like(gaps)
        if dones is not None:
            mask = dones.bool()
            closing[mask] = 0
            self.previous_delta[mask] = 0
        values = torch.cat((original, self.base_action, (gaps/.1).clamp(-5,5),
                            closing.clamp(-5,5), self.previous_delta/self.max_rad), dim=-1)
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError("non-finite residual observation")
        self.previous_gaps = gaps.clone()
        self.cached_observation = TensorDict({'policy': values}, batch_size=[self.num_envs])
        return self.cached_observation

    def get_observations(self):
        if self.cached_observation is None:
            return self._build_observation(self.env.get_observations())
        return self.cached_observation

    def step(self, raw_residual):
        self.get_observations()
        delta = bounded_delta(raw_residual, self.max_rad)
        combined = compose_action(self.base_action, delta, self.action_scale)
        if not bool(torch.isfinite(combined).all()):
            raise RuntimeError("non-finite combined action")
        self.unwrapped._v14_residual_previous.copy_(self.previous_delta)
        self.unwrapped._v14_residual_delta.copy_(delta)
        self.max_observed_delta_rad = max(self.max_observed_delta_rad, float(delta.abs().max()))
        obs, rewards, dones, extras = self.env.step(combined)
        self.previous_delta.copy_(delta)
        return self._build_observation(obs, dones), rewards, dones, extras

    def reset(self):
        obs, extras = self.env.reset()
        self.previous_delta.zero_()
        self.previous_gaps = None
        self.unwrapped._v14_residual_delta.zero_()
        self.unwrapped._v14_residual_previous.zero_()
        return self._build_observation(obs), extras

    def close(self):
        self.env.close()
