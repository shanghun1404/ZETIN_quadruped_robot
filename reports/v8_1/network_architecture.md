# V8.1 network architecture

Identical to V8. `v8_1_ppo_cfg.py` deep-copies `V8_PPO_CFG` and changes only
`experiment_name` / `run_name`, so the network cannot drift from V8 by accident
and any behavioural difference is attributable to the duty-floor reward alone.

## Policy / value network — rsl_rl stock `ActorCritic` (plain MLP, no recurrence)

```
observation (48)  ->  512  ->  256  ->  128  ->  12 actions
                      elu     elu     elu
critic: same shape, same 48-dim input (obs_groups critic = policy)
```

`init_noise_std = 1.0` (learned per-action log-std, published Spot value).
No observation normalisation on either head.

The GRU (`SmoothActorCritic`) used through V5 is deliberately absent: it was a
confound while the gait itself was broken. Reintroducing it is a later isolated
change, not part of V8.1.

## Observation vector (48)

| slice | term | dim |
|---|---|---|
| 0:3   | base linear velocity (body frame)  | 3 |
| 3:6   | base angular velocity (body frame) | 3 |
| 6:9   | projected gravity                  | 3 |
| 9:12  | velocity command                   | 3 |
| 12:24 | joint positions (relative to default) | 12 |
| 24:36 | joint velocities                   | 12 |
| 36:48 | previous action                    | 12 |

No height scan — V8.1 is flat ground. V9 appends 187 scan points for 235 total.

## Action

`JointPositionActionCfg`, all 12 joints, `scale = 0.2`, `use_default_offset=True`.
No action smoothing filter.

## Control rates

`sim.dt = 0.005` (200 Hz physics), `decimation = 4` -> **50 Hz control**.
Episode length 20 s. 4096 parallel environments, `num_steps_per_env = 24`.

## PPO

`clip_param 0.2`, `entropy_coef 0.0025`, `value_loss_coef 0.5`,
`num_learning_epochs 5`, `num_mini_batches 4`, `learning_rate 1e-3` with
adaptive schedule to `desired_kl 0.01`, `gamma 0.99`, `lam 0.95`,
`max_grad_norm 1.0`, `save_interval 100`.
