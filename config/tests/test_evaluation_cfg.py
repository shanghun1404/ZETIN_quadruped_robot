from types import SimpleNamespace

from evaluation_cfg import configure_deterministic_evaluation


def _namespace(**values):
    return SimpleNamespace(**values)


def _fake_env_cfg():
    observations = _namespace(
        **{name: _namespace(enable_corruption=True) for name in ("proprio", "terrain", "command", "critic")}
    )
    command = _namespace(
        resampling_time_range=(1.0, 2.0),
        heading_command=True,
        rel_standing_envs=0.5,
        rel_heading_envs=0.5,
        ranges=_namespace(lin_vel_x=(0.0, 0.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-1.0, 1.0)),
    )
    events = _namespace(
        physics_material=_namespace(params={}),
        add_base_mass=object(),
        base_com=object(),
        base_external_force_torque=object(),
        push_robot=object(),
        reset_base=_namespace(params={"pose_range": {}, "velocity_range": {}}),
        reset_robot_joints=_namespace(params={}),
    )
    return _namespace(
        scene=_namespace(num_envs=1),
        episode_length_s=1.0,
        seed=0,
        curriculum=object(),
        observations=observations,
        commands=_namespace(base_velocity=command),
        events=events,
        actions=_namespace(joint_pos=_namespace(alpha_range=(0.20, 0.35))),
        terminations=_namespace(support_loss=object()),
    )


def test_configure_deterministic_evaluation_fixes_inputs_without_support_termination():
    cfg = _fake_env_cfg()
    configure_deterministic_evaluation(cfg, num_envs=64, semantic_speed=0.40)

    assert cfg.scene.num_envs == 64
    assert cfg.episode_length_s == 60.0
    assert cfg.seed == 42
    assert cfg.curriculum is None
    assert all(not getattr(cfg.observations, name).enable_corruption for name in ("proprio", "terrain", "command", "critic"))
    command = cfg.commands.base_velocity
    assert command.resampling_time_range == (60.0, 60.0)
    assert command.heading_command is False
    assert command.rel_standing_envs == 0.0
    assert command.rel_heading_envs == 0.0
    assert command.ranges.lin_vel_x == (-0.40, -0.40)
    assert command.ranges.lin_vel_y == (0.0, 0.0)
    assert command.ranges.ang_vel_z == (0.0, 0.0)
    assert command.ranges.heading == (0.0, 0.0)
    assert cfg.events.physics_material.params == {
        "static_friction_range": (0.9, 0.9),
        "dynamic_friction_range": (0.7, 0.7),
    }
    for event_name in ("add_base_mass", "base_com", "base_external_force_torque", "push_robot"):
        assert getattr(cfg.events, event_name) is None
    zero_ranges = {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
    assert cfg.events.reset_base.params["pose_range"] == zero_ranges
    assert cfg.events.reset_base.params["velocity_range"] == zero_ranges
    assert cfg.events.reset_robot_joints.params["position_range"] == (1.0, 1.0)
    assert cfg.events.reset_robot_joints.params["velocity_range"] == (0.0, 0.0)
    assert cfg.actions.joint_pos.alpha_range == (0.275, 0.275)
    assert cfg.terminations.support_loss is None
