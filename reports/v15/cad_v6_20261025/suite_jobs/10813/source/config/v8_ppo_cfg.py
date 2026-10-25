"""V8 PPO settings -- the published Spot flat-locomotion agent config.

Differences from V5-V7, all restored to the published values:
  * ``init_noise_std`` 1.0 rather than 0.35 -- V5 explored far too little.
  * ``entropy_coef`` 0.0025 rather than 0.001.
  * ``value_loss_coef`` 0.5 rather than 1.0.
  * A plain MLP ``ActorCritic`` rather than the custom GRU ``SmoothActorCritic``,
    removing the recurrent policy as a confound. It can be reintroduced later as
    an isolated change once a natural gait exists.
  * ``max_iterations`` 20000 rather than 500-4000.
"""

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


V8_PPO_CFG = RslRlOnPolicyRunnerCfg(
    seed=42,
    device="cuda:0",
    num_steps_per_env=24,
    max_iterations=20000,
    obs_groups={"policy": ["policy"], "critic": ["policy"]},
    clip_actions=None,
    save_interval=100,
    experiment_name="guide_v8_spot_recipe",
    run_name="spot_recipe_trot",
    logger="tensorboard",
    policy=RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0025,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
)
