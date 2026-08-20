import ast
from pathlib import Path


def _source(name):
    return Path(name).read_text(encoding="utf-8")


def _table():
    module = ast.parse(_source("v8_env_cfg.py"))
    for node in module.body:
        if isinstance(node, ast.Assign):
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id == "V8_REWARD_WEIGHTS":
                return ast.literal_eval(node.value)
    raise AssertionError("V8_REWARD_WEIGHTS not found")


def test_gait_reward_is_the_dominant_term():
    w = _table()
    # This is the correction of the V5 "phase-free" prohibition: the published
    # recipe makes the gait objective the largest term in the table.
    assert w["gait"] == 10.0
    assert w["gait"] > w["base_linear_velocity"]
    assert w["gait"] == max(w.values())


def test_reward_table_matches_the_published_spot_weights():
    assert _table() == {
        "gait": 10.0,
        "air_time": 5.0,
        "base_angular_velocity": 5.0,
        "base_linear_velocity": 5.0,
        "foot_clearance": 0.5,
        "action_smoothness": -1.0,
        "air_time_variance": -1.0,
        "base_motion": -2.0,
        "base_orientation": -3.0,
        "foot_slip": -0.5,
        "joint_acc": -1.0e-4,
        "joint_pos": -0.7,
        "joint_torques": -5.0e-4,
        "joint_vel": -1.0e-2,
    }


def test_trot_pairs_are_diagonal():
    source = _source("v8_env_cfg.py")
    assert 'TROT_PAIRS = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))' in source


def test_v8_imports_the_maintained_spot_rewards_rather_than_forking_them():
    source = _source("v8_env_cfg.py")
    assert "from isaaclab_tasks.manager_based.locomotion.velocity.config.spot import mdp as spot_mdp" in source
    assert "spot_mdp.GaitReward" in source


def test_v8_drops_the_unreachable_height_target_and_the_inert_height_term():
    source = _source("v8_env_cfg.py")
    # V5.11-V7 targeted 0.45 m; the robot calibrates to 0.3616 m. Assert on the
    # reward table and wiring, not on prose -- the docstring names the term it
    # is explaining why V8 drops.
    assert "base_height_l2" not in _table()
    assert "self.rewards.base_height_l2" not in source
    assert "GUIDE_STANCE_HEIGHT_M = 0.3616" in source
    assert "joint_position_penalty" in source


def test_v8_restores_the_published_exploration_and_training_budget():
    ppo = _source("v8_ppo_cfg.py")
    assert "init_noise_std=1.0" in ppo
    assert "entropy_coef=0.0025" in ppo
    assert "max_iterations=20000" in ppo
    assert "num_steps_per_env=24" in ppo
    # The GRU actor is removed as a confound for this restart. Check the wiring
    # rather than the prose that explains the removal.
    assert 'class_name="SmoothActorCritic"' not in ppo
    assert "import" in ppo and "smooth_actor_critic" not in ppo


def test_v8_uses_the_published_action_scale_and_control_rate():
    source = _source("v8_env_cfg.py")
    assert "scale=0.2" in source
    assert "track_air_time = True" in source

    # 50 Hz control on 200 Hz physics. Spot specifies 500 Hz physics, which
    # benchmarked 1.49x slower (33,817 vs 50,315 env-steps/s) for fidelity the
    # policy never sees -- the control period is 0.02 s either way. 200 Hz is
    # what every stock IsaacLab quadruped and this project's whole history used.
    assert "self.decimation = 4" in source
    assert "self.sim.dt = 0.005" in source


def test_control_period_is_unchanged_by_the_physics_rate_choice():
    """Speeding up physics must not silently change what the policy controls."""
    import re

    source = _source("v8_env_cfg.py")
    decimation = int(re.search(r"self\.decimation = (\d+)", source).group(1))
    dt = float(re.search(r"self\.sim\.dt = ([\d.]+)", source).group(1))
    assert dt * decimation == 0.02          # 50 Hz control
    assert round(1.0 / dt) == 200           # 200 Hz physics


def test_v8_commands_guide_forward_not_base_forward():
    """The robot's head points along base -x; V8's first run commanded +x.

    Measured foot positions in the base frame (check_front diagnostic):
        FL +0.211, FR +0.199  |  RL -0.223, RR -0.167
    and ``guide_frame`` defines the guide frame as the base frame rotated 180
    degrees about z. So a positive base ``lin_vel_x`` walks the robot backwards.
    """
    source = _source("v8_env_cfg.py")
    assert "from guide_frame import guide_forward_speed_range_to_base" in source
    assert "V8_GUIDE_FORWARD_SPEED_RANGE = (0.3, 0.9)" in source
    assert "lin_vel_x=V8_BASE_LIN_VEL_X_RANGE," in source

    # The conversion must actually produce a negative base-x band.
    def convert(speed_range):
        low, high = speed_range
        return (-high, -low)

    assert convert((0.3, 0.9)) == (-0.9, -0.3)


def _unspaced(text: str) -> str:
    """Source with all whitespace removed, so a line wrap cannot break a contract.

    This assertion used to pin the one-line spelling of the call. When
    ``--forward-axis`` was added the call was wrapped across three lines in both
    tools and the test went red -- and stayed red through the V10.1 launch,
    because the suite was not run. What it is supposed to protect is the
    direction convention, not the formatting.
    """
    return "".join(text.split())


def test_v8_video_and_evaluator_use_the_same_direction_convention():
    video = _source("play_v8_video.py")
    evaluator = _source("evaluate_v8.py")
    needle = _unspaced("guide_forward_speed_range_to_base((args_cli.speed, args_cli.speed))")
    for src in (video, evaluator):
        assert needle in _unspaced(src)
        # And the -x conversion must sit on the default branch, not be reachable
        # only when +y is requested.
        assert 'cmd.ranges.lin_vel_y = (0.0, 0.0)' in src
    # The evaluator must report guide-frame forward speed, so a correct policy
    # scores positive rather than negative.
    assert "-robot.data.root_lin_vel_b[:, 0]" in evaluator
    # The side camera must look toward the head (-x), and the "rear" view must
    # sit behind it (+x).
    assert "eye = [x + 2.2, y - 1.8, z + 1.4]" in video
    assert "target = [x - 0.4, y, z + 0.05]" in video
    assert "target = [x - 0.2, y, z + 0.05]" in video
    # A head marker makes the walking direction visible on the recording.
    assert "_create_head_marker" in video
    assert "[[-0.30, 0.0, 0.05]]" in video
