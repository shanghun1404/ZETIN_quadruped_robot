from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


SMOOTH_GUIDE_PPO_CFG = RslRlOnPolicyRunnerCfg(
    seed=42,
    device="cuda:0",
    num_steps_per_env=32,
    max_iterations=4000,
    obs_groups={
        "policy": ["proprio", "terrain", "command"],
        "critic": ["critic"],
    },
    clip_actions=1.0,
    save_interval=100,
    experiment_name="guide_smooth_v5_2_soft_support_gru",
    run_name="height35_soft_support_gru",
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
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    ),
)
