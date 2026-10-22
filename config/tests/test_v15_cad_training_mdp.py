"""CPU behavioral checks for the CAD fine-tuning objective."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "config", ROOT / "ros2/v14_ros2_control"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def objective():
    try:
        return importlib.import_module("v15_cad_training_mdp")
    except ModuleNotFoundError as error:
        if error.name == "v15_cad_training_mdp":
            pytest.fail("CAD training command and reward module is not implemented")
        raise


def test_zero_target_rewards_rest_and_penalizes_drift():
    mdp = objective()
    command = torch.zeros(3, 3)
    velocity = torch.tensor([[0., 0., 0.], [0., .25, 0.], [.25, 0., 0.]])
    yaw = torch.tensor([0., 0., .25])
    planar = mdp.planar_velocity_reward(command, velocity)
    angular = mdp.yaw_velocity_reward(command, yaw)
    cost = mdp.zero_command_motion_cost(command, velocity, yaw)
    torch.testing.assert_close(planar, torch.tensor([1., .36787944, .36787944]))
    torch.testing.assert_close(angular, torch.tensor([1., 1., .36787944]))
    torch.testing.assert_close(cost, torch.tensor([0., 1., 2.]))


def test_reverse_command_is_negative_cad_y_and_matching_reverse_is_best():
    mdp = objective()
    command = mdp.commands_for_categories(torch.tensor([2]), torch.full((1, 6), .5))
    torch.testing.assert_close(command, torch.tensor([[0., -.3, 0.]]))
    forward = torch.tensor([[0., .3, 0.]])
    assert mdp.planar_velocity_reward(command, command).item() == 1.
    assert mdp.planar_velocity_reward(command, forward).item() < .01


def test_lateral_and_pure_yaw_commands_do_not_require_forward_motion():
    mdp = objective()
    command = mdp.commands_for_categories(torch.tensor([3, 4]), torch.full((2, 6), .5))
    torch.testing.assert_close(command, torch.tensor([[.225, 0., 0.], [0., 0., .275]]))
    torch.testing.assert_close(mdp.planar_velocity_reward(command, command), torch.ones(2))
    torch.testing.assert_close(mdp.yaw_velocity_reward(command, command[:, 2]), torch.ones(2))
    torch.testing.assert_close(mdp.zero_command_motion_cost(command, command, command[:, 2]), torch.zeros(2))


def test_balanced_sampling_covers_all_six_motion_categories():
    mdp = objective()
    command, categories = mdp.sample_cad_velocity_commands(
        50_000, device="cpu", generator=torch.Generator().manual_seed(701),
    )
    observed = torch.bincount(categories, minlength=6).float() / len(categories)
    torch.testing.assert_close(observed, torch.tensor([.20, .30, .15, .15, .10, .10]), atol=.008, rtol=0.)
    assert torch.count_nonzero(command[categories == 0]) == 0
    assert torch.all(command[categories == 1, 1] > 0.)
    assert torch.all(command[categories == 2, 1] < 0.)
    assert torch.any(command[categories == 3, 0] < 0.)
    assert torch.any(command[categories == 3, 0] > 0.)
    assert torch.any(command[categories == 4, 2] < 0.)
    assert torch.any(command[categories == 4, 2] > 0.)


def test_command_update_preserves_yaw_and_motion_until_next_resample():
    mdp = objective()
    command = object.__new__(mdp.CadVelocityCommand)
    command.vel_command_b = torch.tensor([[0., 0., -.3], [.2, -.3, .2], [0., 0., 0.]])
    before = command.vel_command_b.clone()
    command._update_command()
    command._update_command()
    torch.testing.assert_close(command.vel_command_b, before)


def test_stair_column_mask_restricts_only_requested_half_of_terrain():
    mdp = objective()
    types = torch.tensor([0, 7, 8, 15])
    mask = mdp.terrain_forward_mask(types, num_cols=16, fraction_start=.5)
    torch.testing.assert_close(mask, torch.tensor([False, False, True, True]))
    commands = mdp.commands_for_categories(
        torch.tensor([4, 2, 4, 2]), torch.full((4, 6), .5), force_forward_mask=mask,
    )
    torch.testing.assert_close(commands, torch.tensor([[0., 0., .275], [0., -.3, 0.], [0., .375, 0.], [0., .375, 0.]]))
    assert not mdp.terrain_forward_mask(types, num_cols=16, fraction_start=None).any()


def test_rated_cost_distinguishes_rms_load_from_excess_rated_load():
    mdp = objective()
    rated = torch.tensor([8.3, 18.])
    torque = torch.tensor([[0., 0.], [8.3, 18.], [16.6, 36.]])
    torch.testing.assert_close(mdp.rated_load_squared(torque, rated), torch.tensor([0., 1., 4.]))
    torch.testing.assert_close(mdp.excess_rated_load_squared(torque, rated), torch.tensor([0., 0., 1.]))


def test_rms_excess_cost_keeps_one_overloaded_motor_visible_for_sustained_load():
    mdp = objective()
    rated = torch.tensor([8.3, 8.3, 18., 18., 8.3, 8.3, 18., 18., 18., 18., 18., 18.])
    torque = torch.zeros(3, 12)
    torque[1, 1] = 9.13  # FR_HAA at 110% of rated torque.
    torque[2, 10] = 19.8  # RR_KFE at 110% of rated torque.
    rms = torque.expand(100, -1, -1).square().mean(0).sqrt()
    torch.testing.assert_close(
        mdp.excess_rms_load_squared(rms, rated, target_fraction=1.),
        torch.tensor([0., .01, .01]),
    )


def test_rms_excess_uses_per_motor_ratings_and_default_headroom():
    mdp = objective()
    rated = torch.tensor([8.3, 8.3, 18., 18., 8.3, 8.3, 18., 18., 18., 18., 18., 18.])
    rms = torch.zeros(3, 12, dtype=torch.float64)
    rms[0, 1], rms[0, 10] = 7.47, 16.2
    rms[1, 1], rms[1, 10] = 8.217, 17.82
    rms[2, 1], rms[2, 10] = 8.217, 16.2
    torch.testing.assert_close(
        mdp.excess_rms_load_squared(rms, rated.to(torch.float64)),
        torch.tensor([0., .02, .01], dtype=torch.float64), atol=1e-12, rtol=1e-5,
    )


def test_rms_motor_reward_reads_policy_rms_and_contract_joint_order():
    mdp = objective()
    rms = torch.zeros(2, 12, dtype=torch.float64)
    rms[0, 1], rms[0, 10] = 9.13, 19.8
    rms[1, 2] = 19.8
    action = SimpleNamespace(estimated_torque_rms_policy=rms, estimated_torque_policy=torch.zeros_like(rms))
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: {"custom_action": action}[name]))
    torch.testing.assert_close(
        mdp.excess_rms_motor_load(env, action_name="custom_action", target_fraction=1.),
        torch.tensor([.02, .01], dtype=torch.float64),
    )
    env.action_manager.get_term = lambda name: {"joint_pos": action}[name]
    rms.zero_()
    torch.testing.assert_close(mdp.excess_rms_motor_load(env), torch.zeros(2, dtype=torch.float64))


@pytest.mark.parametrize("target_fraction", [0., -.1, 1.01, float("nan"), float("inf")])
def test_rms_excess_rejects_invalid_target_fraction(target_fraction):
    mdp = objective()
    with pytest.raises(ValueError, match="fraction"):
        mdp.excess_rms_load_squared(torch.zeros(1, 12), torch.ones(12), target_fraction=target_fraction)


@pytest.mark.parametrize("rms_value", [-1., float("nan"), float("inf")])
def test_rms_excess_rejects_signed_or_nonfinite_rms(rms_value):
    mdp = objective()
    rms = torch.zeros(1, 12)
    rms[0, 1] = rms_value
    with pytest.raises(ValueError, match="RMS"):
        mdp.excess_rms_load_squared(rms, torch.ones(12))


def test_rms_excess_rejects_misaligned_or_invalid_ratings():
    mdp = objective()
    with pytest.raises(ValueError, match="joint"):
        mdp.excess_rms_load_squared(torch.zeros(1, 12), torch.ones(11))
    for bad_rating in (0., -1., float("nan"), float("inf")):
        rated = torch.ones(12)
        rated[0] = bad_rating
        with pytest.raises(ValueError, match="positive"):
            mdp.excess_rms_load_squared(torch.zeros(1, 12), rated)


@pytest.mark.parametrize("kinds,proportions,counts", [
    (("flat", "rough"), (.9, .1), (116, 12)),
    (("flat", "rough", "up", "down"), (.4, .1, .25, .25), (52, 12, 32, 32)),
    (("flat", "up", "down"), (.5, .25, .25), (64, 32, 32)),
])
def test_terrain_kind_mask_matches_native_generator_column_counts(kinds, proportions, counts):
    mdp = objective()
    columns = torch.arange(128)
    masks = [mdp.terrain_kind_mask(columns, num_cols=128, terrain_kinds=kinds,
                                  proportions=proportions, selected_kinds=(kind,)) for kind in kinds]
    assert tuple(int(mask.sum()) for mask in masks) == counts
    torch.testing.assert_close(torch.stack(masks).sum(0), torch.ones(128, dtype=torch.long))


def test_terrain_kind_mask_uses_declared_kind_order_and_normalized_proportions():
    mdp = objective()
    columns = torch.tensor([0, 51, 52, 63, 64, 95, 96, 127])
    torch.testing.assert_close(
        mdp.terrain_kind_mask(columns, num_cols=128, terrain_kinds=("up", "flat", "down", "rough"),
                              proportions=(4., 1., 2.5, 2.5), selected_kinds=("up", "down")),
        torch.tensor([True, True, False, False, True, True, False, False]),
    )


def test_terrain_kind_mask_matches_native_offset_and_strict_cumulative_boundary():
    mdp = objective()
    columns = torch.tensor([398, 399, 400, 998, 999])
    arguments = dict(num_cols=1000, terrain_kinds=("flat", "rough", "up", "down"),
                     proportions=(.4, .1, .25, .25))
    torch.testing.assert_close(mdp.terrain_kind_mask(columns, **arguments, selected_kinds=("flat",)),
                               torch.tensor([True, False, False, False, False]))
    torch.testing.assert_close(mdp.terrain_kind_mask(columns, **arguments, selected_kinds=("rough",)),
                               torch.tensor([False, True, True, False, False]))
    torch.testing.assert_close(mdp.terrain_kind_mask(columns, **arguments, selected_kinds=("down",)),
                               torch.tensor([False, False, False, True, False]))


def test_terrain_kind_mask_empty_selection_and_zero_proportions_select_nothing():
    mdp = objective()
    columns = torch.tensor([0, 63, 64, 127])
    arguments = dict(num_cols=128, terrain_kinds=("rough", "flat", "up"), proportions=(0., 1., 0.))
    assert not mdp.terrain_kind_mask(columns, **arguments, selected_kinds=()).any()
    assert not mdp.terrain_kind_mask(columns, **arguments, selected_kinds=("rough", "up")).any()
    assert mdp.terrain_kind_mask(columns, **arguments, selected_kinds=("flat",)).all()


@pytest.mark.parametrize("num_cols", [0, -1, 1.5, True])
def test_terrain_kind_mask_rejects_invalid_column_counts(num_cols):
    mdp = objective()
    with pytest.raises(ValueError, match="column"):
        mdp.terrain_kind_mask(torch.tensor([0]), num_cols=num_cols, terrain_kinds=("flat",),
                              proportions=(1.,), selected_kinds=())


@pytest.mark.parametrize("kinds,proportions", [
    ((), ()), (("flat", "rough"), (1.,)), (("flat",), (0.,)),
    (("flat",), (-1.,)), (("flat",), (float("nan"),)), (("flat",), (float("inf"),)),
])
def test_terrain_kind_mask_rejects_invalid_kind_proportions(kinds, proportions):
    mdp = objective()
    with pytest.raises(ValueError, match="proportions"):
        mdp.terrain_kind_mask(torch.tensor([0]), num_cols=128, terrain_kinds=kinds,
                              proportions=proportions, selected_kinds=())


def test_rated_policy_limits_follow_names_and_not_cad_joint_order():
    mdp = objective()
    names = ("FL_HAA", "FR_HFE", "RL_KFE", "BR_HAA")
    torch.testing.assert_close(mdp.policy_rated_torque_limits(names), torch.tensor([8.3, 18., 18., 8.3]))
    with pytest.raises(ValueError, match="rated"):
        mdp.policy_rated_torque_limits(("FL_ankle",))


def test_unique_self_contact_pairs_exclude_adjacent_and_fold_both_directions():
    mdp = objective()
    filters = mdp.cad_self_contact_filters()
    pairs = [(row["body"], target) for row in filters for target in row["targets"]]
    assert len(pairs) == 132
    assert ("base", "FL_hip") not in pairs
    assert ("FL_hip", "FL_thigh") not in pairs
    assert ("FL_thigh", "FL_shank") not in pairs
    force = torch.zeros(2, 132, 3)
    i = pairs.index(("FL_thigh", "FR_thigh"))
    j = pairs.index(("FR_thigh", "FL_thigh"))
    force[0, i, 0] = 2.
    force[0, j, 0] = -2.
    force[1, j, 2] = 3.
    norms = mdp.unique_self_contact_force_norms(force, pairs)
    assert norms.shape == (2, 66)
    torch.testing.assert_close((norms > 1.).sum(-1), torch.tensor([1, 1]))
    torch.testing.assert_close(norms.square().sum(-1), torch.tensor([4., 9.]))


def test_self_contact_rewards_use_filtered_pair_force_and_ignore_ground_net_force():
    mdp = objective()
    filters = mdp.cad_self_contact_filters()
    sensors = {}
    for i, row in enumerate(filters):
        sensors[f"pair_{i}"] = SimpleNamespace(data=SimpleNamespace(
            force_matrix_w=torch.zeros(2, 1, len(row["targets"]), 3),
            net_forces_w=torch.full((2, 1, 3), 1000.),
        ))
    env = SimpleNamespace(scene=sensors)
    torch.testing.assert_close(mdp.self_contact_count(env), torch.zeros(2))
    source = next(i for i, row in enumerate(filters) if row["body"] == "FL_thigh")
    target = filters[source]["targets"].index("FR_thigh")
    sensors[f"pair_{source}"].data.force_matrix_w[1, 0, target, 0] = 2.
    torch.testing.assert_close(mdp.self_contact_count(env), torch.tensor([0., 1.]))
    torch.testing.assert_close(mdp.self_contact_force_l2(env, reference_force=1.), torch.tensor([0., 1.]))


def test_pair_force_sensor_without_filters_fails_instead_of_silently_using_net_force():
    mdp = objective()
    env = SimpleNamespace(scene={"pair_0": SimpleNamespace(data=SimpleNamespace(force_matrix_w=None))})
    with pytest.raises(ValueError, match="filtered"):
        mdp.self_contact_count(env)


def test_invalid_distribution_or_rated_limits_fail_early():
    mdp = objective()
    with pytest.raises(ValueError, match="probabilities"):
        mdp.sample_cad_velocity_commands(3, device="cpu", probabilities=(1., 1., 0., 0., 0., 0.))
    with pytest.raises(ValueError, match="positive"):
        mdp.rated_load_squared(torch.ones(1, 2), torch.tensor([8.3, 0.]))


@pytest.mark.parametrize("command,velocity,yaw,expected", [
    ([0., -.3, 0.], [0., -.3, 0.], 0., 1.),
    ([.2, 0., 0.], [.2, 0., 0.], 0., 1.),
    ([0., 0., -.25], [0., 0., 0.], -.25, 1.),
    ([.2, -.3, .25], [0., 0., 0.], 0., 0.),
    ([0., -.3, 0.], [0., .3, 0.], 0., -1.),
    ([.2, 0., 0.], [-.2, 0., 0.], 0., -1.),
    ([0., 0., .25], [0., 0., 0.], -.25, -1.),
    ([0., -.3, 0.], [0., -1.2, 0.], 0., 1.),
    ([.2, 0., 0.], [1.2, 0., 0.], 0., 1.),
    ([0., 0., -.25], [0., 0., 0.], -1., 1.),
    ([.2, -.3, .25], [.1, -.3, 0.], -.25, 1. / 6.),
    ([0., .3, 0.], [123., .15, 0.], 15., .5),
    ([1e-7, -.3, -5e-7], [100., -.3, 0.], 20., 1.),
    ([0., 0., 0.], [123., -123., 0.], 999., 0.),
])
def test_directed_motion_progress_tracks_signed_active_components(command, velocity, yaw, expected):
    mdp = objective()
    actual = mdp.directed_motion_progress(
        torch.tensor([command]), torch.tensor([velocity]), torch.tensor([yaw]),
    )
    torch.testing.assert_close(actual, torch.tensor([expected]))


def test_directed_motion_progress_wrapper_reads_native_velocity_and_command():
    mdp = objective()
    commands = torch.tensor([[0., -.3, 0.], [0., 0., -.25]])
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_command=lambda name: commands),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(
            root_lin_vel_b=torch.tensor([[0., -.15, 0.], [0., 0., 0.]]),
            root_ang_vel_b=torch.tensor([[0., 0., 0.], [0., 0., -.25]]),
        ))},
    )
    torch.testing.assert_close(mdp.directed_motion_progress_reward(env), torch.tensor([.5, 1.]))


def test_command_term_uses_configured_nondefault_motion_distribution():
    mdp = objective()

    class CpuCommand(mdp.CadVelocityCommand):
        @property
        def num_envs(self):
            return self._env.num_envs

        @property
        def device(self):
            return "cpu"

    command = object.__new__(CpuCommand)
    command._env = SimpleNamespace(num_envs=50_000)
    command.cfg = mdp.CadVelocityCommandCfg()
    command.cfg.command_probabilities = (.10, .10, .30, .25, .20, .05)
    command.vel_command_b = torch.zeros(command.num_envs, 3)
    command.motion_category = torch.zeros(command.num_envs, dtype=torch.long)
    command.is_standing_env = torch.zeros(command.num_envs, dtype=torch.bool)
    command.is_heading_env = torch.zeros(command.num_envs, dtype=torch.bool)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(701)
        command._resample_command(torch.arange(command.num_envs))
    observed = torch.bincount(command.motion_category, minlength=6).float() / command.num_envs
    torch.testing.assert_close(observed, torch.tensor([.10, .10, .30, .25, .20, .05]), atol=.008, rtol=0.)
    assert torch.all(command.vel_command_b[command.motion_category == 2, 1] < 0.)
    assert torch.all(command.is_standing_env == (command.motion_category == 0))
    assert not command.is_heading_env.any()
