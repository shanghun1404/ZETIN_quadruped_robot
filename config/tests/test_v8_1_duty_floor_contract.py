"""Contract tests for V8.1: a duty floor under V8's trot, and nothing else.

V8 produced the right gait *phase* and the wrong duty factor (contact fractions
FL 0.161 / FR 0.789 / RL 0.799 / RR 0.162 at 8000 iterations). These tests pin
the properties that make V8.1 a clean single-variable answer to that: the floor
exists and is strong enough to matter, the fourteen Spot weights are untouched,
and the observation/action contract that lets V8's evaluator judge a V8.1
checkpoint still holds.
"""

import ast
from pathlib import Path

import pytest


def _source(name):
    return Path(name).read_text(encoding="utf-8")


def _const(name, module="v8_1_env_cfg.py"):
    tree = ast.parse(_source(module))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {module}")


def _v8_weights():
    tree = ast.parse(_source("v8_env_cfg.py"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == "V8_REWARD_WEIGHTS":
                return ast.literal_eval(node.value)
    raise AssertionError("V8_REWARD_WEIGHTS not found")


def test_v8_1_adds_exactly_one_reward_term():
    """A second simultaneous edit would make the result unattributable."""
    source = _source("v8_1_env_cfg.py")
    assert "**V8_REWARD_WEIGHTS" in source
    assert '"duty_floor": DUTY_FLOOR_WEIGHT' in source
    # Exactly one term is assigned onto the inherited reward group.
    assert source.count("self.rewards.") == 1
    assert "self.rewards.duty_floor = RewardTermCfg(" in source
    # And nothing may reassign a V8 weight on the way past.
    for term in _v8_weights():
        assert f"self.rewards.{term}" not in source


def test_v8_1_inherits_v8_rather_than_restating_it():
    """Inheritance is what guarantees the other fourteen weights cannot drift."""
    source = _source("v8_1_env_cfg.py")
    assert "class GuideDutyFloorEnvCfg(GuideSpotRecipeEnvCfg)" in source
    assert "super().__post_init__()" in source
    # The Spot reward functions must still come from the maintained upstream
    # module, not be forked into this file.
    assert "spot_mdp" not in source


def test_duty_floor_targets_the_measured_failure():
    floor = _const("DUTY_CONTACT_FLOOR")
    # V8 measured 0.161 on its worst foot; the floor has to sit above that or it
    # cannot describe the defect at all.
    assert floor > 0.161
    # It must also sit above the evaluator's 0.30 gate, so training carries
    # margin against the number it is judged by.
    assert floor >= 0.30
    assert floor == 0.35
    # And below a healthy trot's ~0.5 duty factor, so a good gait pays nothing.
    assert floor < 0.5


def test_duty_floor_weight_is_strong_enough_to_move_and_weak_enough_to_lose():
    weight = _const("DUTY_FLOOR_WEIGHT")
    floor = _const("DUTY_CONTACT_FLOOR")
    v8 = _v8_weights()

    assert weight < 0.0, "a floor is a penalty"

    # Magnitude at V8's measured failure, in the same per-second units as the
    # TensorBoard reward rates quoted in the module docstring.
    deficit_at_failure = (floor - 0.161) / floor
    rate = abs(weight) * deficit_at_failure

    # V6's lesson: a term producing 0.022 per step against a 5.0 reward is inert.
    # This must be the largest penalty in the table at the behaviour it targets,
    # or there is no reason to expect it to move anything.
    largest_existing_penalty_rate = 1.267  # action_smoothness, measured at iter 7999
    assert rate > largest_existing_penalty_rate

    # But it must never be able to buy out the gait reward it exists to
    # complement -- even at its bounded worst case of |weight| * 1.0.
    assert abs(weight) < v8["gait"]


def test_duty_floor_is_bounded_so_dying_never_beats_walking():
    """The unbounded-penalty failure mode already killed one experiment here.

    ``worst_foot_contact_deficit`` clamps to [0, 1], so the term cannot exceed
    ``|weight|`` per second no matter how badly the gait degrades, and cannot
    make termination cheaper than continuing.
    """
    math_source = _source("v5_12_math.py")
    assert "def worst_foot_contact_deficit" in math_source
    block = math_source.split("def worst_foot_contact_deficit")[1].split("\ndef ")[0]
    assert "clamp(" in block and "min=0.0" in block and "max=1.0" in block


def test_duty_floor_reuses_the_one_v5_12_term_that_moved_its_metric():
    source = _source("v8_1_env_cfg.py")
    assert "import v5_12_mdp" in source
    assert "func=v5_12_mdp.abandoned_foot_penalty" in source
    # Reused, not reimplemented -- a fresh copy could drift from the version
    # whose behaviour was actually measured.
    assert "def abandoned_foot_penalty" not in source
    assert "class abandoned_foot_penalty" in _source("v5_12_mdp.py")


def test_duty_floor_measures_contact_the_same_way_the_evaluator_reports_it():
    """Train against the quantity that is reported, or the gate means nothing."""
    threshold = _const("DUTY_CONTACT_THRESHOLD")
    assert threshold == 5.0
    # evaluate_v8.py calls a foot loaded above 5.0 N.
    assert "> 5.0" in _source("evaluate_v8.py")
    # The penalty scores the worst foot; the gate scores the worst foot.
    assert "worst = deficit.amax(dim=-1)" in _source("v5_12_math.py")
    assert "min(contact_fraction.values()) >= 0.30" in _source("evaluate_v8.py")


def test_duty_floor_exempts_a_robot_commanded_to_stand_still():
    """Otherwise standing on four planted feet would be penalised as abandonment."""
    source = _source("v8_1_env_cfg.py")
    assert '"moving_threshold": 0.05' in source
    mdp_source = _source("v5_12_mdp.py")
    assert "moving = _moving_mask(env, command_name, moving_threshold)" in mdp_source
    # The comment that used to sit here claimed the mask was "direction-agnostic".
    # It was not: it read one column, so it was sign-agnostic on a single axis,
    # and V10.1 moving the command to +y silenced the term for a whole run while
    # this assertion kept passing. Pin the planar magnitude instead.
    tree = ast.parse(mdp_source)
    mask = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_moving_mask"
    )
    mask_source = ast.get_source_segment(mdp_source, mask)
    assert "torch.linalg.vector_norm(command[:, :2], dim=-1)" in mask_source
    assert "planar_speed >= moving_threshold" in mask_source
    assert "[:, 0]" not in mask_source


def test_duty_window_resolves_a_duty_factor_inside_an_episode():
    window = _const("DUTY_WINDOW_S")
    # Control runs at 50 Hz and episodes last 20 s.
    assert window * 50 >= 25, "window must span several strides"
    assert window <= 20.0 / 4, "window must react well inside an episode"
    assert window == 1.0


def test_v8_1_agent_config_is_v8_deep_copied_not_restated():
    ppo = _source("v8_1_ppo_cfg.py")
    assert "copy.deepcopy(V8_PPO_CFG)" in ppo
    # Only the log destination may differ.
    assert 'V8_1_PPO_CFG.experiment_name = "guide_v8_1_duty_floor"' in ppo
    for forbidden in ("init_noise_std", "actor_hidden_dims", "num_steps_per_env", "learning_rate"):
        assert forbidden not in ppo


def test_v8_evaluator_reuse_is_licensed_by_an_identical_observation_space():
    """V8.1 has no evaluator of its own; that is only sound if obs/actions match."""
    source = _source("v8_1_env_cfg.py")
    for forbidden in ("observations", "actions:", "ObsTerm", "JointPositionActionCfg"):
        assert forbidden not in source
    queue = _source("queue_v8_1_pilot.sh")
    assert "evaluate_v8.py" in queue
    assert "play_v8_video.py" in queue
    # The probe must enforce the precondition rather than leaving it to comment.
    probe = _source("probe_v8_1.py")
    assert 'report["obs_width"] != 48' in probe


def test_probe_proves_the_term_fires_instead_of_assuming_it():
    """V6 shipped a term that was wired up, weighted, and did nothing."""
    probe = _source("probe_v8_1.py")
    assert "term._duty[:, 0] = 0.0" in probe
    assert "fraction_firing" in probe
    assert "duty_penalty_when_planted" in probe
    # Both directions: it must fire when a foot is abandoned and stay silent
    # when all four are loaded.
    assert "did not fire on an abandoned foot" in probe
    assert "penalizes fully loaded feet" in probe
    # Both branches must drive the duty estimate directly. Measuring a settling
    # robot 40 steps after a randomized reset tests the reset transient, not the
    # term -- that mistake made the first version of this probe fail spuriously.
    assert probe.count("term._duty[:] = 1.0") == 2


def test_probe_guards_the_untouched_v8_weights():
    probe = _source("probe_v8_1.py")
    assert "V8 weights drifted" in probe
    assert "for n in V8_REWARD_WEIGHTS" in probe
    assert 'abs(weights["duty_floor"]) >= weights["gait"]' in probe


def test_queue_scores_intermediate_checkpoints_not_only_the_last_one():
    """V8 ended 8000 iterations worse than it was at 1600 and nobody could see it."""
    queue = _source("queue_v8_1_pilot.sh")
    assert "V8_1_SWEEP_ITERS" in queue
    sweep = _const_shell(queue, "SWEEP_ITERS")
    assert sweep, "the sweep must actually name checkpoints"
    # The regression V8 showed lived between 1600 and 8000, so the sweep has to
    # sample inside that span rather than only near the end.
    iters = [int(x) for x in sweep.split()]
    assert min(iters) < 2500
    assert len(iters) >= 3
    assert "v8_1_pilot_sweep.json" in queue
    assert "ranked" in queue


def test_sweep_names_checkpoints_that_are_actually_written():
    """model_1999.pt is never written; asking for it sweeps nothing, silently.

    rsl_rl saves on multiples of ``save_interval`` plus one final
    ``model_<iters-1>``. The first version of this sweep asked for 1999/3999/5999
    and every one was skipped by the file-existence guard, so the ranking
    contained only the final checkpoint -- indistinguishable from a sweep that
    ran and found nothing.
    """
    save_interval = 100
    assert f"save_interval={save_interval}" in _source("v8_ppo_cfg.py")
    queue = _source("queue_v8_1_pilot.sh")
    for it in (int(x) for x in _const_shell(queue, "SWEEP_ITERS").split()):
        assert it % save_interval == 0, f"model_{it}.pt is never written"


def test_sweep_reports_checkpoints_it_could_not_score():
    """A silently empty sweep looks exactly like a sweep that found no problem."""
    queue = _source("queue_v8_1_pilot.sh")
    assert "SWEEP_MISSING" in queue
    assert "sweep_missing_checkpoint" in queue
    assert "sweep_incomplete" in queue
    # The bare `[[ -f "$ckpt" ]] || continue` that swallowed the failure is gone.
    assert '[[ -f "$ckpt" ]] || continue' not in queue


def _const_shell(source, name):
    for line in source.splitlines():
        line = line.strip()
        if line.startswith(f"{name}=") and ":-" in line:
            return line.split(":-", 1)[1].split("}", 1)[0]
    raise AssertionError(f"{name} not found in shell source")


@pytest.mark.parametrize("path", ["v8_1_env_cfg.py", "v8_1_ppo_cfg.py", "train_v8_1.py", "probe_v8_1.py"])
def test_v8_1_sources_parse(path):
    ast.parse(_source(path))
