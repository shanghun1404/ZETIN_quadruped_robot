"""Execute the actual evaluator's configuration branches without starting Isaac."""

import ast
import argparse
import copy
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import pytest


def setup_evaluator(scenario, heading, monkeypatch, control="isaaclab", **overrides):
    source = ast.parse((Path(__file__).parents[1]/"evaluate_v13_torque.py").read_text())
    main = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    first = next(i for i,n in enumerate(main.body) if assigns(n,"env_type"))
    last = next(i for i,n in enumerate(main.body) if assigns(n,"agent_cfg"))
    main.body = main.body[first:last] + [ast.parse("return env_cfg, geometry").body[0]]
    module = ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[]))
    terrain = NS(num_rows=8,num_cols=8,size=(8.,8.),curriculum=True,difficulty_range=(0.,1.),
                 sub_terrains={"stairs":NS(step_height_range=(.05,.20),border_width=1.,platform_width=3.,step_width=.3)})
    def env_type():
        return NS(scene=NS(num_envs=0,robot=NS(spawn=NS(usd_path="")),terrain=NS(terrain_generator=copy.deepcopy(terrain))),
                  observations=NS(policy=NS(enable_corruption=True)),events=NS(push_robot=None,reset_base=NS(params={})),
                  sim=NS(dt=.005),decimation=4,
                  commands=NS(base_velocity=NS(debug_vis=True,ranges=NS())),actions=NS(joint_pos=NS()))
    def lane_command(original):
        result=copy.deepcopy(original)
        result.lane_control=True
        return result
    monkeypatch.setitem(sys.modules,"v14_heading_command", NS(make_lane_command_cfg=lane_command))
    args=NS(variant="v13",num_envs=32,protocol="fixed_v2",usd_path="robot.usda",seed=42,
            enable_self_collisions=False,filter_leg_terrain=False,effort_guard_fraction=None,
            seconds=12.,warmup_seconds=0.,scenario=scenario,terrain_level=5,heading_control=heading,forward_speed=.5,
            control=control,ros2_namespace="/v14",joint_drive="dcmotor")
    vars(args).update(overrides)
    scope=dict(copy=copy,args_cli=args,GuideV13TorqueEnvCfg=env_type,GuideV2ScannerTerrainEnvCfg=env_type,
               STAIRS_UP_EVAL_TERRAIN_CFG=terrain,STAIRS_DOWN_EVAL_TERRAIN_CFG=terrain)
    exec(compile(module,"<evaluator-config>","exec"),scope)
    return scope["main"]()


@pytest.mark.parametrize("scenario",["flat","stairs_up","stairs_down"])
@pytest.mark.parametrize("heading",[False,True])
def test_heading_option_does_not_skip_geometry_or_read_uninitialized_command(scenario,heading,monkeypatch):
    cfg, geometry=setup_evaluator(scenario,heading,monkeypatch)
    assert cfg.commands.base_velocity.ranges.lin_vel_y == (.5,.5)
    assert bool(getattr(cfg.commands.base_velocity,"lane_control",False)) == heading
    if scenario == "flat":
        assert cfg.scene.terrain.terrain_generator is None
    else:
        assert cfg.scene.terrain.terrain_generator.num_cols == 32
        assert cfg.scene.terrain.terrain_generator.num_rows == 1
        assert geometry["step_height_m"] == pytest.approx(.15714285714285717)


def test_legacy_trajectory_rejected_before_simulator_launch(monkeypatch):
    tree=ast.parse((Path(__file__).parents[1]/"evaluate_v13_torque.py").read_text())
    first=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=="parser" for t in n.targets))
    last=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=="simulation_app" for t in n.targets))
    cli=ast.fix_missing_locations(ast.Module(body=tree.body[first:last],type_ignores=[]))
    monkeypatch.setattr(sys,"argv",["eval","--variant","v13","--actor-checkpoint","a",
                         "--student-checkpoint","b","--usd-path","c","--output","r.json",
                         "--trajectory-output","trace.npz","--protocol","legacy"])
    with pytest.raises(SystemExit) as error:
        exec(compile(cli,"<evaluator-cli>","exec"),dict(argparse=argparse,__doc__="evaluation",
             AppLauncher=NS(add_app_launcher_args=lambda parser:None)))
    assert error.value.code == 2


def test_ros2_control_swaps_action_term_and_joint_drives(monkeypatch):
    calls = []
    bridge = NS(V14Ros2JointTargetActionCfg=lambda **kwargs: NS(ros2=kwargs),
                implicit_joint_drives=lambda robot: calls.append(robot))
    monkeypatch.setitem(sys.modules, "v14_ros2_bridge", bridge)
    cfg, _ = setup_evaluator("flat", False, monkeypatch, control="ros2")
    assert cfg.actions.joint_pos.ros2 == {"asset_name": "robot", "namespace": "/v14"}
    assert calls == [cfg.scene.robot]


def test_physx_joint_drive_uses_v15_guarded_action_without_delay(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "v15_env_cfg", NS(
        use_physx_joint_drives=lambda cfg, **kwargs: calls.append(kwargs)))
    setup_evaluator("flat", False, monkeypatch, joint_drive="physx", effort_guard_fraction=.99)
    assert calls == [{"max_delay_steps": 0, "guard_fraction": .99}]
