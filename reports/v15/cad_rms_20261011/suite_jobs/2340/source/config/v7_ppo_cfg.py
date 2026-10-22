from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


# Identical GRU network and optimizer to V5.6-V5.11. V5.12 trains FRESH under the
# V5.11 reference reward plus the two anti-cheat penalties.
#
# 4000 iterations, matching V5.13 so the two runs are directly comparable.
# The reference natural walk was trained for 8000; 4000 is half that and well
# past the point where V5.11's 500-iteration pilot proved nothing.
V7_PPO_CFG = RslRlOnPolicyRunnerCfg(
    seed=42,
    device="cuda:0",
    num_steps_per_env=32,
    max_iterations=4000,
    obs_groups={
        "policy": ["proprio", "terrain", "command"],
        "critic": ["critic"],
    },
    clip_actions=1.0,
    save_interval=50,
    experiment_name="guide_smooth_v7_small_action_gru",
    run_name="height45_speed0510_small_action",
    logger="tensorboard",
    policy=RslRlPpoActorCriticCfg(
        class_name="SmoothActorCritic",
        init_noise_std=0.35,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
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
