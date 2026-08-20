"""Contract tests for V9.1: a termination set that rough terrain can survive.

V9 was stopped at iteration 4142 of 8000 with 82.5% of episodes ending on body
contact, a mean episode length of 93 steps out of 1000, and a terrain curriculum
that had not moved off its starting level in 4142 iterations. These tests pin the
properties that make V9.1 a clean single-variable answer to that.
"""

import ast
from pathlib import Path

import pytest


def _source(name):
    return Path(name).read_text(encoding="utf-8")


def _const(name, module="v9_1_env_cfg.py"):
    tree = ast.parse(_source(module))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {module}")


def test_v9_1_stops_terminating_on_the_swinging_links():
    """A thigh or calf against a stair edge is locomotion, not a crash."""
    bodies = _const("TERMINATION_BODIES")
    assert "thigh" not in bodies
    assert "calf" not in bodies
    # The two that must still end the episode: the trunk and the body-mounted hip.
    assert "base_link" in bodies
    assert "hip" in bodies


def test_v9_1_matches_what_stock_isaaclab_terminates_on_for_rough_terrain():
    """IsaacLab's own rough-terrain config ends episodes on body contact only.

    V8 added hip/thigh/calf as hardware protection, which costs nothing on flat
    ground and pins the curriculum on terrain.
    """
    bodies = _const("TERMINATION_BODIES")
    # V9.1 keeps the hip beyond stock, and nothing else.
    extra = [part for part in bodies.split("|") if "base" not in part]
    assert extra == [".*_hip"]


def test_v9_1_changes_only_the_termination_set():
    source = _source("v9_1_env_cfg.py")
    assert "class GuideTerrainFixedEnvCfg(GuideTerrainEnvCfg)" in source
    assert "super().__post_init__()" in source
    # Exactly one assignment onto the inherited config.
    assert source.count("self.terminations.") == 1
    assert "self.rewards" not in source
    assert "self.scene" not in source
    assert "self.observations" not in source
    assert "self.curriculum" not in source
    assert "self.actions" not in source
    assert "V9_1_REWARD_WEIGHTS = dict(V9_REWARD_WEIGHTS)" in source


def test_v9_1_leaves_the_contact_threshold_alone():
    """Raising it would be a second change and would blunt the base test too.

    The first version of this test contained ``assert "..." or True``, which can
    never fail: it would have stayed green through any threshold change at all.
    Parse the DoneTerm instead and require it to reference the named constants,
    so disconnecting either one breaks the test.
    """
    assert _const("TERMINATION_THRESHOLD_N") == 1.0

    tree = ast.parse(_source("v9_1_env_cfg.py"))
    assigns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "body_contact" for t in node.targets
        )
    ]
    assert len(assigns) == 1, "exactly one body_contact termination must be assigned"
    wired = ast.unparse(assigns[0])
    assert "TERMINATION_BODIES" in wired
    assert "TERMINATION_THRESHOLD_N" in wired
    assert "mdp.illegal_contact" in wired
    # A literal would silently decouple the constant the tests above check.
    assert "1.0" not in wired.split("threshold")[1][:40]


def test_v9_1_keeps_the_duty_floor_through_two_levels_of_inheritance():
    """V8.1's floor is the reason V9.1 has a balanced gait to preserve at all."""
    assert "from v9_env_cfg import GuideTerrainEnvCfg, V9_REWARD_WEIGHTS" in _source("v9_1_env_cfg.py")
    assert "V9_REWARD_WEIGHTS = dict(V8_1_REWARD_WEIGHTS)" in _source("v9_env_cfg.py")
    assert '"duty_floor": DUTY_FLOOR_WEIGHT' in _source("v8_1_env_cfg.py")
    # And the probe must confirm it survived, not assume it.
    assert "duty_floor lost in inheritance" in _source("probe_v9_1.py")


def test_probe_proves_the_claim_by_driving_contact_forces():
    """A regex that resolves to the wrong bodies passes a source read and still fails."""
    probe = _source("probe_v9_1.py")
    assert "fires_when_loaded" in probe
    assert "net_forces_w_history" in probe
    for key in ("fires_on_thigh_calf", "fires_on_base_link", "fires_on_hip", "fires_on_feet"):
        assert key in probe
    # Both directions, plus the regression that would end every episode.
    assert 'report["fires_on_thigh_calf"] != 0.0' in probe
    assert 'report["fires_on_base_link"] != 1.0' in probe
    assert 'report["fires_on_feet"] != 0.0' in probe


def test_evaluation_runs_under_the_config_the_checkpoint_trained_with():
    """Survival is a gate, and the termination set decides it.

    A V9.1 checkpoint scored under V9's stricter set would be penalised for
    contacts its own training deliberately tolerates, so every terrain version
    must remain selectable rather than the tooling tracking only the newest one.
    """
    for name in ("evaluate_v9.py", "play_v9_video.py"):
        source = _source(name)
        assert '"--env"' in source
        assert '"v9_1"' in source, "V9.1 checkpoints must stay evaluable"
        assert "from v9_1_env_cfg import GuideTerrainFixedEnvCfg" in source
        # The cfg must actually be used, not just imported.
        assert "GuideTerrainEnvCfg()" not in source
    queue = _source("queue_v9_1_pilot.sh")
    assert queue.count("--env v9_1") >= 6   # two gates + four videos
    assert '--env "$VER"' in _source("sweep_v9_checkpoints.sh")


def test_v9_1_established_that_the_prediction_must_be_checked_early():
    """V9 wasted 2h43m proving a curriculum that was stuck by iteration 500.

    V9.1 introduced the early check; V9.2 then showed a single threshold at a
    single iteration was not enough, because V9.1's own curriculum passed the
    check at 0.9065 while already falling from a 1.584 peak. The trend-based
    replacement and its regression test live in
    ``test_v9_2_command_band_contract``; what V9.1 owns is that the check exists
    at all and runs inside its queue.
    """
    queue = _source("queue_v9_1_pilot.sh")
    assert "check_terrain_progress.py" in queue
    assert "v9_1_terrain_progress.json" in queue
    # It must run beside training rather than after it, or it is not early.
    assert "TRAIN_PID" in queue
    assert 'kill -0 "$TRAIN_PID"' in queue


def test_v9_1_agent_config_is_v9_deep_copied_not_restated():
    ppo = _source("v9_1_ppo_cfg.py")
    assert "copy.deepcopy(V9_PPO_CFG)" in ppo
    assert 'V9_1_PPO_CFG.experiment_name = "guide_v9_1_terrain"' in ppo
    for forbidden in ("init_noise_std", "actor_hidden_dims", "num_steps_per_env", "learning_rate"):
        assert forbidden not in ppo


@pytest.mark.parametrize(
    "path",
    ["v9_1_env_cfg.py", "v9_1_ppo_cfg.py", "train_v9_1.py", "probe_v9_1.py", "check_terrain_progress.py"],
)
def test_v9_1_sources_parse(path):
    ast.parse(_source(path))
