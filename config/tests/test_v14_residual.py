"""Residual safety contracts; CPU tensors, no physics simulation."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from v14_clearance import interleg_clearances_m
from v14_residual_math import bounded_delta, capsule_gaps, clearance_cost, compose_action
from v14_residual_policy import ResidualPolicyWrapper


NAMES = tuple(f"{leg}_{part}_profile" for leg in ("FL", "FR", "RL", "RR") for part in ("thigh", "calf"))


def test_tensor_gap_matches_independent_numpy_geometry():
    rng = np.random.default_rng(912)
    positions = rng.normal(size=(30, 8, 3)) * .2
    rotations = rng.normal(size=(30, 8, 4))
    rotations[0] = (1., 0., 0., 0.)  # Parallel capsules, including coincident pair.
    positions[0, 2] = positions[0, 0]
    actual = capsule_gaps(torch.tensor(positions), torch.tensor(rotations))
    expected = interleg_clearances_m(positions, rotations, NAMES)
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-10)
    assert actual[0, 0].item() == pytest.approx(-.04)
    np.testing.assert_allclose(capsule_gaps(torch.tensor(positions).float(), torch.tensor(rotations).float()), expected, atol=1e-5)


def test_zero_and_noisy_residual_are_bounded_in_radians():
    base = torch.randn(7, 12)
    zero = bounded_delta(torch.zeros_like(base), .05)
    assert torch.equal(compose_action(base, zero, .2), base)
    raw = torch.linspace(-1e6, 1e6, 84).reshape(7, 12)
    delta = bounded_delta(raw, .05)
    assert delta.abs().max() <= .05
    torch.testing.assert_close((compose_action(base, delta, .2)-base)*.2, delta)
    for invalid in (0., -.05, float('nan')):
        with pytest.raises(ValueError):
            bounded_delta(raw, invalid)
    with pytest.raises(ValueError):
        compose_action(base, delta, 0.)


def test_clearance_cost_has_no_cost_far_away_and_penalizes_overlap():
    gaps = torch.ones(4, 24)
    gaps[1, 7] = .01
    gaps[2, 9] = 0.
    gaps[3, 10] = -.04
    torch.testing.assert_close(clearance_cost(gaps), torch.tensor([0., .25, 1., 9.]))


class BaseActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.ones(12) * .2)

    def act_inference(self, obs):
        assert obs['policy'].shape[1] == 235
        return obs['policy'][:, :12] + self.bias


class TensorEnv:
    """Only replaces the Isaac physics boundary; real wrapper/math remain active."""
    def __init__(self):
        self.num_envs, self.num_actions, self.device = 2, 12, 'cpu'
        self.max_episode_length = 1000
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.step_dt = .02
        self.unwrapped = self
        self.obs = torch.zeros(2, 235)
        self.cfg = SimpleNamespace(actions=SimpleNamespace(joint_pos=SimpleNamespace(scale=.2)))
        positions = torch.zeros(2, 8, 3)
        positions[:, :, 1] = torch.arange(8)*.08
        rotations = torch.zeros(2, 8, 4); rotations[:, :, 0] = 1
        self.robot = SimpleNamespace(body_names=NAMES, data=SimpleNamespace(body_pos_w=positions, body_quat_w=rotations))
        self.scene = {'robot': self.robot}
        self.done_next = torch.zeros(2, dtype=torch.long)

    def get_observations(self):
        return TensorDict({'policy': self.obs.clone()}, batch_size=[2])

    def step(self, action):
        self.obs[:, :12] = action
        self.robot.data.body_pos_w[:, 2, 1] -= .01
        self.obs[self.done_next.bool()] = 0
        self.robot.data.body_pos_w[self.done_next.bool(), 2, 1] = .16
        return self.get_observations(), torch.ones(2), self.done_next.clone(), {}

    def reset(self):
        self.obs.zero_()
        return self.get_observations(), {}


def test_wrapper_keeps_combined_history_and_freezes_base():
    base, raw = BaseActor(), TensorEnv()
    env = ResidualPolicyWrapper(raw, base)
    obs = env.get_observations()
    assert obs['policy'].shape == (2, 307)
    assert not base.training and not any(p.requires_grad for p in base.parameters())
    assert torch.equal(env.get_observations()['policy'], obs['policy'])
    next_obs, _, _, _ = env.step(torch.zeros(2, 12))
    torch.testing.assert_close(next_obs['policy'][:, :12], torch.full((2, 12), .2))
    torch.testing.assert_close(next_obs['policy'][:, 235:247], torch.full((2, 12), .4))
    raw.episode_length_buf.zero_()
    env.episode_length_buf = torch.tensor([30, 40])
    assert raw.episode_length_buf.tolist() == [30, 40]


def test_async_reset_clears_gap_rate_and_previous_correction_only_for_reset_env():
    raw = TensorEnv(); env = ResidualPolicyWrapper(raw, BaseActor())
    env.get_observations()
    raw.done_next[0] = 1
    obs, _, _, _ = env.step(torch.ones(2, 12))
    assert torch.count_nonzero(obs['policy'][0, 271:307]) == 0
    assert torch.count_nonzero(obs['policy'][1, 271:295]) > 0
    assert torch.count_nonzero(obs['policy'][1, 295:307]) == 12
    again = env.get_observations()
    assert torch.equal(again['policy'], obs['policy'])
    reset, _ = env.reset()
    assert torch.count_nonzero(reset['policy'][:, 271:307]) == 0


def test_residual_starts_at_zero_and_optimizer_cannot_change_frozen_base():
    from rsl_rl.modules import ActorCritic
    from v14_residual_policy import initialize_residual_actor, module_digest
    base = BaseActor(); env = ResidualPolicyWrapper(TensorEnv(), base)
    obs = env.get_observations()
    policy = ActorCritic(obs, {'policy':['policy'], 'critic':['policy']}, 12,
                         actor_hidden_dims=[128,128], critic_hidden_dims=[128,128], init_noise_std=.05)
    initialize_residual_actor(policy)
    assert torch.count_nonzero(policy.act_inference(obs)) == 0
    before = module_digest(base)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss = (policy.act_inference(obs)-.1).square().mean()
    loss.backward(); optimizer.step()
    assert torch.count_nonzero(policy.act_inference(obs)) > 0
    assert module_digest(base) == before


@pytest.mark.parametrize('key,value', [('base_sha256','wrong'),('student_sha256','wrong'),('usd_sha256','wrong'),
                         ('schema',2),('max_rad',float('nan')),('max_rad',.08),('heading_control',False),
                         ('action_scale',.5),('checkpoint_bytes','different residual weights')])
def test_checkpoint_binding_rejects_wrong_assets_and_unsupported_schema(tmp_path,key,value):
    import json
    from v14_residual_policy import file_digest, load_residual_manifest
    base, student, usd, checkpoint = [tmp_path/name for name in ('base.pt','student.pt','robot.usda','model_0.pt')]
    for p in (base,student,usd,checkpoint): p.write_text(p.name)
    manifest = {'schema':1, 'base_sha256':file_digest(base), 'student_sha256':file_digest(student),
                'usd_sha256':file_digest(usd), 'max_rad':.05, 'heading_control':True,
                'observation_dim':307, 'actor_hidden_dims':[128,128], 'action_scale':.2,
                'checkpoints':{checkpoint.name:file_digest(checkpoint)}}
    sidecar=tmp_path/'v14_residual_manifest.json'
    sidecar.write_text(json.dumps(manifest))
    args=dict(base_checkpoint=base, student_checkpoint=student, usd_path=usd, heading_control=True)
    assert load_residual_manifest(checkpoint, **args)['max_rad'] == .05
    if key == 'checkpoint_bytes':
        checkpoint.write_text(value)
    else:
        sidecar.write_text(json.dumps({**manifest,key:value}))
    with pytest.raises(ValueError): load_residual_manifest(checkpoint, **args)


def test_initial_checkpoint_saves_with_installed_runner_before_learn(tmp_path):
    from rsl_rl.runners import OnPolicyRunner
    from v14_residual_policy import save_initial_residual
    cfg = dict(num_steps_per_env=2, save_interval=50, obs_groups={'policy':['policy'],'critic':['policy']},
               logger='tensorboard', policy=dict(class_name='ActorCritic',actor_hidden_dims=[8],critic_hidden_dims=[8]),
               algorithm=dict(class_name='PPO',num_learning_epochs=1,num_mini_batches=1))
    env = ResidualPolicyWrapper(TensorEnv(), BaseActor())
    runner = OnPolicyRunner(env,cfg,log_dir=str(tmp_path),device='cpu')
    checkpoint=tmp_path/'model_zero.pt'
    save_initial_residual(runner,checkpoint)
    loaded=torch.load(checkpoint,weights_only=False)
    assert all(torch.isfinite(v).all() for v in loaded['model_state_dict'].values())
    assert torch.count_nonzero(runner.alg.policy.act_inference(env.get_observations())) == 0
    if runner.writer is not None: runner.writer.close()


def test_resume_keeps_trained_policy_optimizer_and_advances_completed_iteration(tmp_path):
    """Catch accidental zero initialization, optimizer reset, or repeated update IDs."""
    from rsl_rl.runners import OnPolicyRunner
    import v14_residual_policy as residual
    cfg = dict(num_steps_per_env=2, save_interval=50, obs_groups={'policy':['policy'],'critic':['policy']},
               logger='tensorboard', policy=dict(class_name='ActorCritic',actor_hidden_dims=[8],critic_hidden_dims=[8]),
               algorithm=dict(class_name='PPO',num_learning_epochs=1,num_mini_batches=1))
    env = ResidualPolicyWrapper(TensorEnv(), BaseActor())
    import copy
    source = OnPolicyRunner(env,copy.deepcopy(cfg),log_dir=str(tmp_path/'source'),device='cpu')
    obs = env.get_observations()
    source.alg.optimizer.zero_grad()
    loss = (source.alg.policy.act_inference(obs)-.2).square().mean() + source.alg.policy.evaluate(obs).square().mean()
    loss.backward()
    source.alg.optimizer.step()
    source.current_learning_iteration = 100
    source._prepare_logging_writer()
    checkpoint = tmp_path/'trained.pt'
    source.save(str(checkpoint))
    wanted = source.alg.policy.act_inference(obs).detach().clone()
    assert torch.count_nonzero(wanted) > 0
    target = OnPolicyRunner(env,copy.deepcopy(cfg),log_dir=str(tmp_path/'target'),device='cpu')
    assert hasattr(residual, 'resume_residual'), 'trained residual resume is missing'
    info = residual.resume_residual(target,checkpoint,tmp_path/'model_resume.pt')
    torch.testing.assert_close(target.alg.policy.act_inference(obs), wanted, rtol=0, atol=0)
    assert residual.module_digest(target.alg.policy) == residual.module_digest(source.alg.policy)
    assert target.current_learning_iteration == info['next_iteration'] == 101
    assert info['loaded_iteration'] == 100
    actual = target.alg.optimizer.state_dict()
    expected = source.alg.optimizer.state_dict()
    assert actual['param_groups'] == expected['param_groups']
    assert actual['state'].keys() == expected['state'].keys()
    for key in expected['state']:
        for field, value in expected['state'][key].items():
            torch.testing.assert_close(actual['state'][key][field], value, rtol=0, atol=0)
    saved = torch.load(tmp_path/'model_resume.pt',weights_only=False)
    assert saved['iter'] == 100  # Copy describes a completed update, not the next one.
    for runner in (source,target):
        if runner.writer is not None: runner.writer.close()


def test_periodic_checkpoint_is_sealed_and_loadable_before_training_finishes(tmp_path):
    """Catch intermediate checkpoints being unusable until learn() returns."""
    import json
    import v14_residual_policy as residual
    assert hasattr(residual, 'ResidualOnPolicyRunner'), 'periodic checkpoint sealing is missing'
    cfg = dict(num_steps_per_env=2, save_interval=50, obs_groups={'policy':['policy'],'critic':['policy']},
               logger='tensorboard', policy=dict(class_name='ActorCritic',actor_hidden_dims=[8],critic_hidden_dims=[8]),
               algorithm=dict(class_name='PPO',num_learning_epochs=1,num_mini_batches=1))
    base,student,usd = [tmp_path/name for name in ('base.pt','student.pt','robot.usda')]
    for path in (base,student,usd): path.write_text(path.name)
    manifest = dict(schema=1,observation_dim=307,actor_hidden_dims=[128,128],action_scale=.2,
                    heading_control=True,max_rad=.05,base_sha256=residual.file_digest(base),
                    student_sha256=residual.file_digest(student),usd_sha256=residual.file_digest(usd))
    (tmp_path/'v14_residual_manifest.json').write_text(json.dumps(manifest))
    env = ResidualPolicyWrapper(TensorEnv(),BaseActor())
    runner = residual.ResidualOnPolicyRunner(env,cfg,log_dir=str(tmp_path),device='cpu')
    runner._prepare_logging_writer()
    runner.current_learning_iteration = 200
    checkpoint=tmp_path/'model_200.pt'
    runner.save(str(checkpoint))
    residual.load_residual_manifest(checkpoint,base_checkpoint=base,student_checkpoint=student,
                                    usd_path=usd,heading_control=True)
    saved=torch.load(checkpoint,weights_only=False)
    assert saved['iter'] == 200
    assert not list(tmp_path.glob('*.tmp'))
    if runner.writer is not None: runner.writer.close()
