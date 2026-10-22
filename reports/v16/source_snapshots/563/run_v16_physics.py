"""Independent V16 direct-effort controller and population learning on PhysX.

Run only on HPC through launch_v16_physics.sbatch. No RL policy is loaded.
PhysX generalized rigid dynamics are the plant oracle; contact allocation is a
regularized least-squares approximation, not a constrained MPC guarantee.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("probe", "eval", "train"), default="probe")
parser.add_argument("--controller", choices=("pd", "model"), default="model")
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--generations", type=int, default=60)
parser.add_argument("--seconds", type=float, default=6.)
parser.add_argument("--speed", type=float, default=.25)
parser.add_argument("--walk-axis", choices=("x", "y"), default="y", help="source fore-aft direction is +Y; +X retained for diagnostic comparison")
parser.add_argument("--seed", type=int, default=51)
parser.add_argument("--params")
parser.add_argument("--resume", help="CEM distribution/RNG checkpoint with unchanged rollout objective")
parser.add_argument("--output-root", required=True)
parser.add_argument("--usd-path", required=True)
parser.add_argument("--control-decimation", type=int, default=4)
parser.add_argument("--video", action="store_true")
parser.add_argument("--validate-armature", action="store_true", help="noncontact, zero-friction forward-dynamics characterization before probe")
parser.add_argument("--contact-projection", action="store_true", help="project desired acceleration onto stance foot COM anchor constraints; approximate contact kinematics")
parser.add_argument("--gait-mode", choices=("periodic", "prepared_crawl"), default="periodic")
parser.add_argument("--stance-point", choices=("com", "ground", "measured"), default="ground", help="ground: rigid touchdown proxy; measured: SDK mean contact patch point in BOTH acceleration and force maps")
parser.add_argument("--effort-fraction", type=float, default=.90)
parser.add_argument("--joint-friction", type=float, default=.3, help="installed Isaac Sim >=5 static joint-friction effort; setting zero alone leaves USD dynamic/viscous values unchanged")
parser.add_argument("--joint-dynamic-friction", type=float, help="installed Isaac Sim >=5 dynamic Coulomb effort; default preserves USD value")
parser.add_argument("--joint-viscous-friction", type=float, help="installed Isaac Sim >=5 viscous friction coefficient; default preserves USD value")
parser.add_argument("--force-allocator", choices=("ls", "qp"), default="ls", help="qp: equality/unilateral/friction/torque constraints with finite-iteration ADMM diagnostics; force_scale fixed at 1")
parser.add_argument("--base-acceleration-weight", type=float, default=10., help="base task deviation weight versus joint weight1 in the contact-acceleration projection")
parser.add_argument("--smooth-com-reference", action="store_true", help="latched minimum-jerk COM waypoint, reference speed <=.30m/s and acceleration <=1.2m/s² with analytic feedforward")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from guide_robot_v2_cfg import GUIDE_ROBOT_V2_CFG
from v14_terrain_filter import author_v14_ground_filter
from v16_controller_math import (JOINT_NAMES, PARAM_NAMES, LOWER, UPPER, DEFAULT,
                                 digital_acceleration_gains, cem_update, candidate_score,
                                 swing_reference, guard_activity, mask_failed_metric, resume_cem, joint_permutation,
                                 add_joint_armature, forward_metrics, rigid_point_jacobian)
from v16_crawl_math import (contact_acceleration_projection, triangle_signed_margin,
                            triangle_com_target, prepared_crawl_transition, contact_support_masks)
from v16_force_qp import solve_force_qp
from v16_com_reference_math import minimum_jerk_duration, minimum_jerk_reference

DT = .005
LEGS = ("FL", "FR", "RL", "RR")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def skew(v):
    out = torch.zeros((*v.shape[:-1], 3, 3), device=v.device, dtype=v.dtype)
    out[..., 0, 1], out[..., 0, 2] = -v[..., 2], v[..., 1]
    out[..., 1, 0], out[..., 1, 2] = v[..., 2], -v[..., 0]
    out[..., 2, 0], out[..., 2, 1] = -v[..., 1], v[..., 0]
    return out


class Plant:
    def __init__(self):
        self.out = Path(args.output_root)
        self.out.mkdir(parents=True, exist_ok=False)
        self.n = args.num_envs
        self.forward_axis = 0 if args.walk_axis == "x" else 1
        self.lateral_axis = 1 - self.forward_axis
        self.device = args.device
        self.control_dt = DT * args.control_decimation
        self.sim = SimulationContext(SimulationCfg(dt=DT, device=args.device, render_interval=4,
                                                   gravity=(0., 0., -9.81)))
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/ground", ground)
        from pxr import UsdPhysics
        import omni.usd
        ground_filters = [str(prim.GetPath()) for prim in omni.usd.get_context().get_stage().Traverse()
                          if str(prim.GetPath()).startswith("/World/ground") and prim.HasAPI(UsdPhysics.CollisionAPI)]
        if not ground_filters:
            raise RuntimeError("ground collision prim required for contact-point characterization")
        light = sim_utils.DomeLightCfg(intensity=1500.)
        light.func("/World/light", light)
        cfg = copy.deepcopy(GUIDE_ROBOT_V2_CFG)
        cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        cfg.spawn.usd_path = args.usd_path
        cfg.spawn.articulation_props.enabled_self_collisions = True
        cfg.spawn.articulation_props.solver_position_iteration_count = 8
        cfg.spawn.articulation_props.solver_velocity_iteration_count = 1
        cfg.init_state.pos = (0., 0., .46)
        cfg.actuators = {
            "haa": IdealPDActuatorCfg(joint_names_expr=[".*_HAA"], stiffness=0., damping=0.,
                                      effort_limit=24.8, effort_limit_sim=24.8,
                                      velocity_limit_sim=50., friction=args.joint_friction,
                                      dynamic_friction=args.joint_dynamic_friction, viscous_friction=args.joint_viscous_friction),
            "hfe_kfe": IdealPDActuatorCfg(joint_names_expr=[".*_HFE", ".*_KFE"],
                                          stiffness=0., damping=0., effort_limit=48.,
                                          effort_limit_sim=48., velocity_limit_sim=50., friction=args.joint_friction,
                                          dynamic_friction=args.joint_dynamic_friction, viscous_friction=args.joint_viscous_friction),
        }

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            robot: ArticulationCfg = cfg
            feet: ContactSensorCfg = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base/.*_foot", update_period=0., history_length=1)
            contact_fl: ContactSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/base/FL_foot", filter_prim_paths_expr=ground_filters, track_contact_points=True, track_friction_forces=True)
            contact_fr: ContactSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/base/FR_foot", filter_prim_paths_expr=ground_filters, track_contact_points=True, track_friction_forces=True)
            contact_rl: ContactSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/base/RL_foot", filter_prim_paths_expr=ground_filters, track_contact_points=True, track_friction_forces=True)
            contact_rr: ContactSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/base/RR_foot", filter_prim_paths_expr=ground_filters, track_contact_points=True, track_friction_forces=True)

        self.scene = InteractiveScene(SceneCfg(num_envs=self.n, env_spacing=3., replicate_physics=False))
        author_v14_ground_filter(self.scene.stage, self.scene.env_prim_paths)
        self.robot = self.scene["robot"]
        self.sim.reset()
        self.scene.reset()
        self.view = self.robot.root_physx_view
        self.joint_ids, self.generalized_ids = joint_permutation(self.robot.joint_names)
        if self.view.shared_metatype.fixed_base:
            raise RuntimeError("V16 plant must have a floating base")
        self.foot_ids = [self.robot.body_names.index(f"{leg}_foot") for leg in LEGS]
        self.sensor_foot_ids = [self.scene["feet"].body_names.index(f"{leg}_foot") for leg in LEGS]
        self.q0 = self.robot.data.default_joint_pos[:, self.joint_ids].clone()
        self.limits = torch.tensor([24.8 if n.endswith("HAA") else 48. for n in JOINT_NAMES],
                                   device=self.device)
        self.eye6 = torch.eye(6, device=self.device)[None]
        self.eye12 = torch.eye(12, device=self.device)[None]
        self.body_masses = self.view.get_masses().to(self.device)
        self.mass = self.body_masses.sum(dim=1)
        self.jacobian_origin = "unvalidated"
        self.previous = torch.zeros((self.n, 12), device=self.device)
        self.reset(51)
        # A short explicit PD settle is initialization, never part of scored rollout.
        for _ in range(400):
            tau = 250. * (self.q0 - self.q()) - 8. * self.dq()
            self.advance(tau)
        self.settled_q = self.q().clone()
        self.settled_root = self.robot.data.root_state_w.clone()
        self.nominal_feet_b = quat_apply_inverse(
            self.robot.data.root_quat_w[:, None, :].expand(-1, 4, -1).reshape(-1, 4),
            (self.robot.data.body_com_pos_w[:, self.foot_ids] -
             self.robot.data.root_com_pos_w[:, None, :]).reshape(-1, 3)).reshape(self.n, 4, 3)
        self.height = (self.robot.data.root_com_pos_w[:, 2] - self.scene.env_origins[:, 2]).clone()
        M, h, J = self.dynamics()
        if M.shape != (self.n, 18, 18) or h.shape != (self.n, 18) or J.shape != (self.n, 4, 6, 18):
            raise RuntimeError(f"generalized dynamics shape mismatch {M.shape}, {h.shape}, {J.shape}")
        # Oracle zero-twist base translational mass must represent all link masses.
        if not torch.allclose(M[:, :3, :3].diagonal(dim1=-2, dim2=-1),
                              self.mass[:, None].expand(-1, 3), atol=.05, rtol=.001):
            raise RuntimeError("generalized mass matrix root coordinates not verified")
        if float(torch.linalg.eigvalsh(M).min()) <= 0:
            raise RuntimeError("generalized physical inertia is not positive definite")
        manifest = {
            "mode": args.mode, "controller": args.controller, "num_envs": self.n,
            "seed": args.seed, "usd_path": args.usd_path, "usd_sha256": sha(args.usd_path),
            "walk_axis_world": args.walk_axis,
            "contact_projection": args.contact_projection,
            "gait_mode": args.gait_mode,
            "stance_point": args.stance_point,
            "base_acceleration_weight": args.base_acceleration_weight,
            "smooth_com_reference": args.smooth_com_reference,
            "com_reference_bounds": {"speed_m_s": .30, "acceleration_m_s2": 1.2, "min_duration_s": .2},
            "joint_friction": args.joint_friction,
            "joint_dynamic_friction": args.joint_dynamic_friction,
            "joint_viscous_friction": args.joint_viscous_friction,
            "measured_joint_friction": self.robot.data.joint_friction_coeff[0, self.joint_ids].cpu().tolist(),
            "measured_joint_dynamic_friction": self.robot.data.joint_dynamic_friction_coeff[0, self.joint_ids].cpu().tolist(),
            "measured_joint_viscous_friction": self.robot.data.joint_viscous_friction_coeff[0, self.joint_ids].cpu().tolist(),
            "joint_friction_convention": "installed Isaac Sim 5.1 uses static/dynamic friction effort and viscous coefficient via get/set_dof_friction_properties; differs from pre-5.0 dimensionless friction model; excluded by M/h oracle",
            "code_sha256": {p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name("v16_controller_math.py"), Path(__file__).with_name("v16_crawl_math.py"), Path(__file__).with_name("v16_force_qp.py"), Path(__file__).with_name("v16_com_reference_math.py"))},
            "actuated_dofs": 12, "floating_base_velocity_dofs": 6,
            "physics_dt_s": DT, "control_dt_s": self.control_dt,
            "input_hold": "torque ZOH between digital ticks; emergency envelope projection each physics step",
            "effort_measurement": "explicit requested/guarded actuation command; solver contact/friction wrench is separate",
            "effort_fraction": args.effort_fraction, "joint_order": list(JOINT_NAMES),
            "physics_joint_order": self.robot.joint_names, "canonical_to_physics_joint_indices": self.joint_ids,
            "parameter_names": list(PARAM_NAMES), "lower": LOWER.tolist(), "upper": UPPER.tolist(),
            "initial_parameters_sha256": sha(args.params) if args.params else None,
            "rigid_body_mass_kg": self.mass.cpu().tolist(), "body_names": self.robot.body_names,
            "contact_sensor_body_names": self.scene["feet"].body_names,
            "ground_contact_filters": ground_filters,
            "contact_force_trace": "legacy contact_force_w is net NORMAL force only; ground_normal_force_w and ground_friction_force_w use canonical order and ground-only filters",
            "measured_point_convention": "SDK mean contact patch point per ground filter, normal-force-weighted across filters; world coordinates; NaN without contact",
            "moment_normalization_length_m": .30,
            "physics_tensor_extension": "omni.physics.tensors 107.3.26 (installed SDK)",
            "generalized_coordinates": "root COM world linear/angular twist first, canonical joint velocities last",
            "tensor_inertia_convention": "at link COM, expressed in rigid-body-prim frame; column-major 9 entries",
            "tensor_com_convention": "principal-frame pose relative to rigid-body prim; position and quaternion xyzw",
            "body_pose_convention": "world position and quaternion wxyz",
            "sdk_reference": "https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html",
            "model_compensation_excludes": ["contact", "joint friction", "damping"],
            "controller_inertia": "raw generalized rigid tensor plus measured joint armature diagonal (comparison 526)",
            "force_allocator": args.force_allocator,
            "force_allocator_details": "qp: fixed 80-iteration ADMM, warm start, equality base wrench, conservative friction pyramid and torque bounds; numerical diagnostics, no stability/infeasibility certificate; force_scale fixed 1" if args.force_allocator == "qp" else "regularized base-wrench least squares with unilateral/friction projection; not a constrained QP",
            "qp_infeasible_policy": "retain finite approximate allocation with diagnostics, then independent motor envelope/slew guard; no claim that infeasible requests satisfy constraints",
            "contact_point_approximation": "ground mode: initial/touchdown flat-plane COM projection latched in foot COM frame and rotated rigidly; same point in acceleration and force maps; not measured mesh contact patch",
            "stance_acceleration_constraint": "declared stance-point anchors; finite-difference Jdot*v; gains 100/s² and 20/s; declared base deviation weight versus joint1; approximate mesh contact kinematics",
            "trace_time_convention": "time_s is post-physics-step state; control_time_s is the input/model evaluation time",
            "acceleration_diagnostic": "5ms finite-difference generalized COM-world twist and joint velocities; predictor uses latest control-tick M+h+ground-point J and post-step ground normal+friction forces; residual also includes contact-point approximation, solver discretization and other contacts",
            "prepared_crawl_gates": {"normal_contact_min_n": 20., "target_inset_m": .055,
                                     "force_constraint_contact_presence_min_n": 1.,
                                     "liftoff_whole_com_margin_m": .04, "whole_com_speed_max_m_s": .06,
                                     "whole_com_target_error_max_m": .015,
                                     "readiness_hold_s": .1, "contact_scope": "foot net-force proxy, not mesh contact patch"},
            "gpu_environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS")},
            "base_actuation": False,
        }
        self.controller_code_sha256 = manifest["code_sha256"]
        dump(self.out / "manifest.json", manifest)
        np.savez_compressed(self.out / "physx_nominal.npz", M=M[0].cpu().numpy(), h=h[0].cpu().numpy(),
                            gravity=self.view.get_gravity_compensation_forces()[0, self.generalized_ids].cpu().numpy(),
                            coriolis=self.view.get_coriolis_and_centrifugal_compensation_forces()[0, self.generalized_ids].cpu().numpy(),
                            J_feet=J[0].cpu().numpy(), q=self.q()[0].cpu().numpy(),
                            dq=self.dq()[0].cpu().numpy(),
                            root_actor_position=self.robot.data.root_pos_w[0].cpu().numpy(),
                            root_quaternion_wxyz=self.robot.data.root_quat_w[0].cpu().numpy(),
                            root_com_position=self.robot.data.root_com_pos_w[0].cpu().numpy(),
                            root_com_quaternion_wxyz=self.robot.data.root_com_quat_w[0].cpu().numpy(),
                            root_actor_twist_w=self.robot.data.root_link_vel_w[0].cpu().numpy(),
                            root_com_linear_velocity=self.robot.data.root_lin_vel_w[0].cpu().numpy(),
                            root_angular_velocity=self.robot.data.root_ang_vel_w[0].cpu().numpy(),
                            body_com_positions=self.robot.data.body_com_pos_w[0].cpu().numpy(),
                            body_com_quaternions_wxyz=self.robot.data.body_com_quat_w[0].cpu().numpy(),
                            body_actor_twists_w=self.robot.data.body_link_vel_w[0].cpu().numpy(),
                            body_com_twists_w=self.robot.data.body_com_vel_w[0].cpu().numpy(),
                            joint_armatures=self.robot.data.joint_armature[0, self.joint_ids].cpu().numpy(),
                            joint_damping=self.robot.data.joint_damping[0, self.joint_ids].cpu().numpy(),
                            joint_friction=self.robot.data.joint_friction_coeff[0, self.joint_ids].cpu().numpy(),
                            gravity_world=np.array([0., 0., -9.81]),
                            env_origin=self.scene.env_origins[0].cpu().numpy(),
                            masses=self.view.get_masses()[0].cpu().numpy(),
                            inertias=self.view.get_inertias()[0].cpu().numpy(),
                            coms=self.view.get_coms()[0].cpu().numpy(),
                            body_positions=self.robot.data.body_link_pos_w[0].cpu().numpy(),
                            body_quaternions=self.robot.data.body_link_quat_w[0].cpu().numpy(),
                            joint_names=np.array(JOINT_NAMES), body_names=np.array(self.robot.body_names))
        self.validate_jacobian()
        if args.validate_armature:
            self.validate_forward_dynamics()

    def reset(self, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        root = self.robot.data.default_root_state.clone()
        root[:, :3] += self.scene.env_origins
        # Small holdout disturbances are separate from policy/gait randomness.
        root[:, 7:10] = .015 * torch.randn((self.n, 3), generator=generator, device=self.device)
        self.robot.write_root_state_to_sim(root)
        self.robot.write_joint_state_to_sim(self.q0, torch.zeros_like(self.q0), joint_ids=self.joint_ids)
        self.robot.set_joint_effort_target(torch.zeros_like(self.q0), joint_ids=self.joint_ids)
        self.scene.reset()
        self.previous.zero_()
        self.sim.forward()
        self.scene.update(DT)

    def advance(self, requested):
        speed = torch.clamp(self.dq() / 50., -1., 1.)
        upper = args.effort_fraction * self.limits * torch.minimum(1. - speed, torch.ones_like(speed))
        lower = args.effort_fraction * self.limits * torch.maximum(-1. - speed, -torch.ones_like(speed))
        # Digital command already has its slew bound; only emergency projection occurs here.
        tau = torch.maximum(torch.minimum(requested, upper), lower)
        tau = torch.nan_to_num(tau, nan=0., posinf=0., neginf=0.)
        self.robot.set_joint_effort_target(tau, joint_ids=self.joint_ids)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(DT)
        self.previous.copy_(tau)
        return tau

    def dynamics(self):
        M = self.view.get_generalized_mass_matrices()[:, self.generalized_ids][:, :, self.generalized_ids]
        h = (self.view.get_gravity_compensation_forces()
             + self.view.get_coriolis_and_centrifugal_compensation_forces())[:, self.generalized_ids]
        J = self.view.get_jacobians()[:, self.foot_ids][..., self.generalized_ids]
        return M, h, J

    def q(self):
        return self.robot.data.joint_pos[:, self.joint_ids]

    def dq(self):
        return self.robot.data.joint_vel[:, self.joint_ids]

    def validate_jacobian(self):
        # A real-velocity characterization identifies which body origin the SDK Jacobian uses.
        root = self.robot.data.root_state_w.clone()
        q = self.q().clone()
        dq = torch.linspace(-.25, .25, 12, device=self.device)[None].repeat(self.n, 1)
        self.robot.write_joint_velocity_to_sim(dq, joint_ids=self.joint_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros((self.n, 6), device=self.device))
        self.sim.forward()
        self.scene.update(DT)
        J = self.view.get_jacobians()[:, self.foot_ids][..., self.generalized_ids]
        predicted = torch.einsum("nfij,nj->nfi", J[..., :3, 6:], dq)
        actor_error = float((predicted - self.robot.data.body_link_lin_vel_w[:, self.foot_ids]).abs().max())
        com_error = float((predicted - self.robot.data.body_com_lin_vel_w[:, self.foot_ids]).abs().max())
        self.jacobian_origin = "com" if com_error < actor_error else "actor"
        if min(actor_error, com_error) > .03:
            raise RuntimeError(f"Cannot characterize Jacobian origin actor={actor_error}, com={com_error}")
        dump(self.out / "jacobian_validation.json", {"actor_velocity_max_error_m_s": actor_error,
            "com_velocity_max_error_m_s": com_error, "selected_origin": self.jacobian_origin,
            "test_joint_velocity_rad_s": dq[0].cpu().tolist()})
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=self.joint_ids)
        self.robot.write_root_state_to_sim(root)
        self.sim.forward()
        self.scene.update(DT)

    def point_jacobian(self, J):
        if self.jacobian_origin == "actor":
            displacement = (self.robot.data.body_com_pos_w[:, self.foot_ids]
                            - self.robot.data.body_link_pos_w[:, self.foot_ids])
            return J[..., :3, :] - skew(displacement) @ J[..., 3:, :]
        return J[..., :3, :]

    def contact_details(self):
        points, normals, frictions = [], [], []
        for leg in LEGS:
            data = self.scene[f"contact_{leg.lower()}"].data
            positions = data.contact_pos_w[:, 0]
            normal = data.force_matrix_w[:, 0]
            weights = torch.where(torch.isfinite(positions).all(dim=-1), normal[..., 2].clamp_min(0.), 0.)
            total = weights.sum(dim=1)
            average = (torch.nan_to_num(positions) * weights[..., None]).sum(dim=1) / total[:, None].clamp_min(1e-6)
            points.append(torch.where((total > 1e-6)[:, None], average, torch.full_like(average, float("nan"))))
            normals.append(normal.sum(dim=1))
            frictions.append(data.friction_forces_w[:, 0].sum(dim=1))
        return torch.stack(points, dim=1), torch.stack(normals, dim=1), torch.stack(frictions, dim=1)

    def validate_forward_dynamics(self):
        # Isolate inertia from contact and friction at zero initial velocity.
        saved = {name: getattr(self.robot.data, name)[:, self.joint_ids].clone() for name in
                 ("joint_friction_coeff", "joint_dynamic_friction_coeff", "joint_viscous_friction_coeff")}
        writers = {"joint_friction_coeff": self.robot.write_joint_friction_coefficient_to_sim,
                   "joint_dynamic_friction_coeff": self.robot.write_joint_dynamic_friction_coefficient_to_sim,
                   "joint_viscous_friction_coeff": self.robot.write_joint_viscous_friction_coefficient_to_sim}
        try:
            for name, value in saved.items():
                writers[name](torch.zeros_like(value), joint_ids=self.joint_ids)
            root = self.robot.data.default_root_state.clone()
            root[:, :3] += self.scene.env_origins
            root[:, 2] += 2.
            root[:, 7:] = 0.
            self.robot.write_root_state_to_sim(root)
            self.robot.write_joint_state_to_sim(self.q0, torch.zeros_like(self.q0), joint_ids=self.joint_ids)
            self.scene.reset()
            self.sim.forward()
            self.scene.update(DT)
            M, h, _ = self.dynamics()
            torque = torch.zeros_like(self.q0)
            for i in range(self.n):
                torque[i, (0, 1, 2, 11)[i % 4]] = 2.
            force = torch.cat((torch.zeros((self.n, 6), device=self.device), torque), dim=1) - h
            rotor = self.robot.data.joint_armature[:, self.joint_ids]
            predicted_rigid = torch.linalg.solve(M, force[..., None]).squeeze(-1)[:, 6:]
            predicted_effective = torch.linalg.solve(add_joint_armature(M, rotor), force[..., None]).squeeze(-1)[:, 6:]
            self.advance(torque)
            observed = self.dq() / DT
            scale = torch.linalg.vector_norm(observed, dim=1).clamp_min(1e-6)
            errors = {"rigid_relative_error": (torch.linalg.vector_norm(predicted_rigid - observed, dim=1) / scale).cpu().tolist(),
                      "armature_relative_error": (torch.linalg.vector_norm(predicted_effective - observed, dim=1) / scale).cpu().tolist(),
                      "predicted_rigid_rad_s2": predicted_rigid.cpu().tolist(),
                      "predicted_with_armature_rad_s2": predicted_effective.cpu().tolist(),
                      "observed_rad_s2": observed.cpu().tolist(),
                      "foot_contact_force_max_n": float(self.scene["feet"].data.net_forces_w.abs().max()),
                      "physics_dt_s": DT, "joint_order": list(JOINT_NAMES),
                      "conditions": "one physics step; suspended by initial position, not an actuator; zero initial velocity; joint friction disabled and restored"}
            dump(self.out / "forward_dynamics_validation.json", errors)
            if max(errors["armature_relative_error"]) > .03 or errors["foot_contact_force_max_n"] > 1e-3:
                raise RuntimeError("armature-inclusive forward dynamics did not match the isolated solver response")
        finally:
            for name, value in saved.items():
                writers[name](value, joint_ids=self.joint_ids)
            self.reset(args.seed)

    def rollout(self, params_np, seed, *, trace=False, stand=False):
        self.reset(seed)
        p = torch.as_tensor(params_np, dtype=torch.float32, device=self.device)
        if p.shape != (self.n, len(PARAM_NAMES)):
            raise ValueError("parameter population shape mismatch")
        if not np.isfinite(params_np).all() or np.any(params_np < LOWER) or np.any(params_np > UPPER):
            raise ValueError("parameters must satisfy the current single-swing controller bounds")
        # Warmup on the scored controller, not an invisible success rescue.
        alive = torch.ones(self.n, dtype=torch.bool, device=self.device)
        initial_xy = self.robot.data.root_com_pos_w[:, :2].clone()
        initial_whole_com_xy = ((self.robot.data.body_com_pos_w * self.body_masses[..., None]).sum(dim=1) / self.mass[:, None])[:, :2].clone()
        qref = self.q0.clone()
        qref_prev = qref.clone()
        kp_kd = np.array([digital_acceleration_gains(float(row[0]), float(row[1]), self.control_dt)
                          for row in params_np])
        kp = torch.as_tensor(kp_kd[:, 0], device=self.device, dtype=torch.float32)
        kd = torch.as_tensor(kp_kd[:, 1], device=self.device, dtype=torch.float32)
        contact_anchor = self.robot.data.body_com_pos_w[:, self.foot_ids].clone()
        foot_quat = self.robot.data.body_com_quat_w[:, self.foot_ids]
        plane_offset = torch.zeros_like(contact_anchor)
        plane_offset[..., 2] = self.scene.env_origins[:, 2, None] - contact_anchor[..., 2]
        contact_point_local = quat_apply_inverse(foot_quat.reshape(-1, 4), plane_offset.reshape(-1, 3)).reshape(self.n, 4, 3)
        point_anchor = contact_anchor + plane_offset
        previous_presence_contact = torch.zeros((self.n, 4), device=self.device, dtype=torch.bool)
        swing_start = contact_anchor.clone()
        swing_end = contact_anchor.clone()
        previous_stance = torch.ones((self.n, 4), dtype=torch.bool, device=self.device)
        previous_foot_jacobian = None
        force_qp_state = None
        force_qp_previous_stance = None
        acceleration_weight = torch.diag(torch.tensor([args.base_acceleration_weight] * 6 + [1.] * 12, device=self.device))
        crawl_stage = torch.zeros(self.n, device=self.device, dtype=torch.long)
        crawl_elapsed = torch.zeros(self.n, device=self.device)
        crawl_ready_elapsed = torch.zeros(self.n, device=self.device)
        crawl_leg_cursor = torch.zeros(self.n, device=self.device, dtype=torch.long)
        crawl_leg_order = torch.tensor([0, 3, 1, 2], device=self.device)
        triangle_ids = torch.tensor([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]], device=self.device)
        environment_ids = torch.arange(self.n, device=self.device)
        liftoff_count = torch.zeros(self.n, device=self.device, dtype=torch.long)
        completed_step_count = torch.zeros(self.n, device=self.device, dtype=torch.long)
        com_reference_step = torch.full_like(completed_step_count, -1)
        com_reference_start = initial_whole_com_xy.clone()
        com_reference_goal = initial_whole_com_xy.clone()
        com_reference_start_time = torch.zeros(self.n, device=self.device)
        com_reference_duration = torch.ones(self.n, device=self.device)
        requested = torch.zeros_like(self.q0)
        raw_requested = requested.clone()
        last_safe_progress = torch.zeros(self.n, device=self.device)
        sums = {k: torch.zeros(self.n, device=self.device) for k in
                ("alive_steps", "speed", "pressure", "saturation", "power", "base_force_residual",
                 "base_moment_residual", "model_samples", "guard_fraction", "stance_acceleration_residual", "stance_foot_speed")}
        qp_ticks = torch.zeros(self.n, device=self.device)
        qp_failed_ticks = torch.zeros_like(qp_ticks)
        qp_max_violation = torch.zeros_like(qp_ticks)
        qp_max_torque_violation = torch.zeros_like(qp_ticks)
        qp_max_base_violation = torch.zeros_like(qp_ticks)
        records = {k: [] for k in ("time_s", "torque_requested_nm", "torque_command_nm", "joint_position_rad",
                                   "joint_velocity_rad_s", "root_position_m", "root_quaternion_wxyz", "alive", "contact_force_w",
                                   "feet_com_position_m", "whole_com_position_m", "scheduled_stance",
                                   "stance_acceleration_residual", "stance_foot_speed_m_s",
                                   "crawl_stage", "swing_leg", "measured_support", "support_triangle_margin_m", "whole_com_target_xy_m")}
        records.update({k: [] for k in ("control_time_s", "contact_point_position_m", "allocated_contact_force_w",
                                        "desired_base_acceleration", "projected_base_acceleration", "base_wrench_residual",
                                        "measured_contact_point_position_m", "ground_normal_force_w", "ground_friction_force_w",
                                        "root_com_velocity_w", "generalized_acceleration_measured_5ms",
                                        "generalized_acceleration_predicted_from_ground_forces_5ms", "generalized_force_unmodeled_5ms",
                                        "force_qp_feasible", "force_qp_constraint_violation")})
        records.update({k: [] for k in ("whole_com_reference_xy_m", "whole_com_reference_velocity_m_s", "whole_com_reference_acceleration_m_s2")})
        video = None
        if args.video and trace:
            from v14_video import RolloutVideos
            video = RolloutVideos(self.sim, self.robot, self.out / f"video_seed{seed}", [0], view="side", fps=25.)
        steps = round(args.seconds / DT)
        measured_steps = 0
        for step in range(steps):
            t = step * DT
            finite_state = (torch.isfinite(self.robot.data.root_state_w).all(dim=1)
                            & torch.isfinite(self.q()).all(dim=1)
                            & torch.isfinite(self.dq()).all(dim=1))
            alive &= finite_state
            # Failed environments are administratively quarantined, never restored to alive.
            # This prevents one invalid rigid-body state poisoning a batch linear solve.
            dead_ids = (~alive).nonzero(as_tuple=False).flatten()
            if dead_ids.numel():
                safe_root = self.robot.data.default_root_state[dead_ids].clone()
                safe_root[:, :3] += self.scene.env_origins[dead_ids]
                safe_root[:, 7:] = 0.
                self.robot.write_root_state_to_sim(safe_root, env_ids=dead_ids)
                self.robot.write_joint_state_to_sim(self.q0[dead_ids], torch.zeros_like(self.q0[dead_ids]),
                                                   joint_ids=self.joint_ids, env_ids=dead_ids)
                self.robot.reset(dead_ids)
                self.sim.forward()
                self.scene.update(DT)
            if step % args.control_decimation == 0:
                control_time = t
                M, h, Jfull = self.dynamics()
                M = add_joint_armature(M, self.robot.data.joint_armature[:, self.joint_ids])
                J = self.point_jacobian(Jfull)
                pos = self.robot.data.root_com_pos_w
                quat = self.robot.data.root_quat_w
                body_vel = self.robot.data.root_lin_vel_w
                angular = self.robot.data.root_ang_vel_w
                feet = self.robot.data.body_com_pos_w[:, self.foot_ids]
                whole_com = (self.robot.data.body_com_pos_w * self.body_masses[..., None]).sum(dim=1) / self.mass[:, None]
                whole_com_velocity = (self.robot.data.body_com_lin_vel_w * self.body_masses[..., None]).sum(dim=1) / self.mass[:, None]
                normal_contact_force = self.scene["feet"].data.net_forces_w[:, self.sensor_foot_ids, 2]
                measured_points, ground_normal, ground_friction = self.contact_details()
                patch_valid = torch.isfinite(measured_points).all(dim=-1) & (ground_normal[..., 2] > 1.)
                if args.stance_point == "measured":
                    normal_contact_force = ground_normal[..., 2]
                physical_contact, measured_contact = contact_support_masks(normal_contact_force)
                swing_leg = crawl_leg_order[crawl_leg_cursor]
                com_target_xy = whole_com[:, :2].clone()
                com_reference_xy = com_target_xy.clone()
                com_reference_velocity = torch.zeros_like(com_target_xy)
                com_reference_acceleration = torch.zeros_like(com_target_xy)
                support_margin = torch.zeros(self.n, device=self.device)
                # Crawl phase (four quarter-offset legs) rather than unsupported diagonal gait.
                offset = torch.tensor([0., .5, .75, .25], device=self.device)
                phase = torch.remainder(torch.clamp(max(0., t - 1.) / p[:, 6, None] - offset[None], min=0.), 1.)
                stance = (phase < p[:, 9, None]) | (t < 1.) | stand
                swing_u = torch.clamp((phase - p[:, 9, None]) / (1 - p[:, 9, None]), 0., 1.)
                if args.gait_mode == "prepared_crawl" and not stand and t >= 1.:
                    control_com_goal_step = completed_step_count.clone()
                    preview_offset = quat_apply(self.robot.data.body_com_quat_w[:, self.foot_ids].reshape(-1, 4),
                                                contact_point_local.reshape(-1, 3)).reshape(self.n, 4, 3)
                    support_points = feet + preview_offset if args.stance_point != "com" else feet
                    if args.stance_point == "measured":
                        support_points = torch.where(patch_valid[..., None], measured_points, support_points)
                    triangle = support_points[environment_ids[:, None], triangle_ids[swing_leg], :2]
                    support_margin, triangle_valid = triangle_signed_margin(triangle, whole_com[:, :2])
                    desired_com_xy = initial_whole_com_xy.clone()
                    # Fixed goal during each gated step; waiting does not move the goal away.
                    desired_com_xy[:, self.forward_axis] += args.speed * p[:, 6] * (completed_step_count + 1) / 4.
                    com_target_xy, target_valid = triangle_com_target(triangle, point_xy=desired_com_xy, inset=.055)
                    com_target_xy = torch.where(target_valid[:, None], com_target_xy, whole_com[:, :2])
                    ready = (alive & triangle_valid & target_valid & (support_margin >= .04)
                             & (torch.linalg.vector_norm(whole_com_velocity[:, :2], dim=1) < .06)
                             & (torch.linalg.vector_norm(com_target_xy - whole_com[:, :2], dim=1) < .015)
                             & (torch.linalg.vector_norm(angular, dim=1) < .5))
                    if args.smooth_com_reference:
                        reference_finished = ((com_reference_step == control_com_goal_step)
                            & (t - com_reference_start_time >= com_reference_duration))
                        ready &= reference_finished
                    supporting_contacts = measured_contact.clone()
                    supporting_contacts[environment_ids, swing_leg] = True
                    crawl_stage, crawl_elapsed, crawl_ready_elapsed, lift, touchdown, next_leg = prepared_crawl_transition(
                        crawl_stage, crawl_elapsed, crawl_ready_elapsed, dt=self.control_dt,
                        ready=ready, all_contacts=measured_contact.all(dim=1) & alive,
                        stance_contacts_ok=supporting_contacts.all(dim=1) & alive,
                        swing_seconds=(1. - p[:, 9]) * p[:, 6])
                    liftoff_count += lift
                    completed_step_count += next_leg
                    crawl_leg_cursor = torch.remainder(crawl_leg_cursor + next_leg.long(), 4)
                    swing_leg = crawl_leg_order[crawl_leg_cursor]
                    stance = torch.ones((self.n, 4), device=self.device, dtype=torch.bool)
                    stance[environment_ids, swing_leg] = crawl_stage != 1
                    swing_u = torch.zeros((self.n, 4), device=self.device)
                    swing_u[environment_ids, swing_leg] = torch.clamp(crawl_elapsed / ((1. - p[:, 9]) * p[:, 6]), 0., 1.)
                    if args.smooth_com_reference:
                        # This goal belongs to the leg/step at entry to this tick.
                        # FSM completion below selects the next triangle on the following tick.
                        new_waypoint = control_com_goal_step != com_reference_step
                        unfinished_handoff = (new_waypoint & alive & (com_reference_step >= 0)
                            & (t - com_reference_start_time < com_reference_duration))
                        if bool(unfinished_handoff.any()):
                            raise RuntimeError("COM waypoint changed before previous segment finished")
                        com_reference_start = torch.where(new_waypoint[:, None],
                            torch.where((com_reference_step < 0)[:, None], whole_com[:, :2], com_reference_goal), com_reference_start)
                        com_reference_goal = torch.where(new_waypoint[:, None], com_target_xy, com_reference_goal)
                        new_duration = minimum_jerk_duration(com_reference_start, com_reference_goal, vmax=.30, amax=1.2)
                        com_reference_duration = torch.where(new_waypoint, new_duration, com_reference_duration)
                        com_reference_start_time = torch.where(new_waypoint, torch.full_like(com_reference_start_time, t), com_reference_start_time)
                        com_reference_step = control_com_goal_step
                        com_reference_xy, com_reference_velocity, com_reference_acceleration = minimum_jerk_reference(
                            com_reference_start, com_reference_goal, t - com_reference_start_time, com_reference_duration)
                    else:
                        com_reference_xy = com_target_xy
                if bool((stance.sum(dim=1) < 3).any()):
                    raise RuntimeError("single-swing crawl requires at least three scheduled stance feet")
                new_contact = stance & ~previous_stance
                new_swing = ~stance & previous_stance
                contact_anchor = torch.where(new_contact[..., None], swing_end, contact_anchor)
                if t < 1.:
                    contact_anchor = feet.clone()
                plane_offset = torch.zeros_like(feet)
                plane_offset[..., 2] = self.scene.env_origins[:, 2, None] - feet[..., 2]
                foot_quat = self.robot.data.body_com_quat_w[:, self.foot_ids]
                touchdown_offset = plane_offset
                if args.stance_point == "measured":
                    touchdown_offset = torch.where(patch_valid[..., None], measured_points - feet, plane_offset)
                fresh_local = quat_apply_inverse(foot_quat.reshape(-1, 4), touchdown_offset.reshape(-1, 3)).reshape(self.n, 4, 3)
                refresh_point = (physical_contact & ~previous_presence_contact) | new_contact | (t < 1.)
                contact_point_local = torch.where(refresh_point[..., None], fresh_local, contact_point_local)
                point_anchor = torch.where(refresh_point[..., None], feet + touchdown_offset, point_anchor)
                rigid_offset = quat_apply(foot_quat.reshape(-1, 4), contact_point_local.reshape(-1, 3)).reshape(self.n, 4, 3)
                contact_points = feet + rigid_offset
                if args.stance_point == "measured":
                    rigid_offset = torch.where(patch_valid[..., None], measured_points - feet, rigid_offset)
                    contact_points = feet + rigid_offset
                Jcontact = rigid_point_jacobian(J, Jfull[..., 3:, :], rigid_offset if args.stance_point != "com" else plane_offset)
                nominal = self.nominal_feet_b.clone()
                # Endpoints are latched at liftoff along the declared world command axis.
                landing = pos[:, None] + quat_apply(
                    quat[:, None].expand(-1, 4, -1).reshape(-1, 4), nominal.reshape(-1, 3)).reshape(self.n, 4, 3)
                landing[..., self.forward_axis] += p[:, 7, None] * (1. if args.gait_mode == "prepared_crawl" else .5)
                landing[..., 2] = contact_anchor[..., 2]
                swing_start = torch.where(new_swing[..., None], contact_anchor, swing_start)
                swing_end = torch.where(new_swing[..., None], landing, swing_end)
                desired_feet = swing_reference(swing_start, swing_end, swing_u, p[:, 8, None])
                desired_feet = torch.where(stance[..., None], contact_anchor, desired_feet)
                if args.stance_point != "com":
                    desired_feet = torch.where((stance & physical_contact)[..., None], point_anchor - rigid_offset, desired_feet)
                if args.gait_mode == "prepared_crawl" and not stand and t >= 1.:
                    search = (crawl_stage == 2) & ~measured_contact[environment_ids, swing_leg]
                    desired_feet[environment_ids, swing_leg, 2] -= search * torch.clamp(crawl_elapsed * .02, max=.03)
                if stand or t < 1.:
                    qref.copy_(self.q0)
                else:
                    error = (desired_feet - feet).reshape(self.n, 12)
                    Jj = J[..., 6:].reshape(self.n, 12, 12)
                    delta = torch.linalg.solve(Jj.transpose(1, 2) @ Jj + .002 * self.eye12,
                                               (Jj.transpose(1, 2) @ error[..., None])).squeeze(-1)
                    qref = self.q() + torch.clamp(delta, -.12, .12)
                    posture_range = torch.tensor([.4, .9, .9] * 4, device=self.device)
                    qref = torch.maximum(torch.minimum(qref, self.q0 + posture_range), self.q0 - posture_range)
                dqref = torch.clamp((qref - qref_prev) / self.control_dt, -3., 3.)
                qref_prev = qref.clone()
                aj = kp[:, None] * (qref - self.q()) + kd[:, None] * (dqref - self.dq())
                aj += p[:, 11, None] * kp[:, None] * (self.q0 - self.q())
                aj = torch.clamp(aj, -150., 150.)
                base_a = torch.zeros((self.n, 6), device=self.device)
                desired_speed = 0. if stand or t < 1. else args.speed
                base_a[:, self.forward_axis] = p[:, 3] * (desired_speed - body_vel[:, self.forward_axis])
                base_a[:, self.lateral_axis] = p[:, 2] * (initial_xy[:, self.lateral_axis] - pos[:, self.lateral_axis]) - p[:, 3] * body_vel[:, self.lateral_axis]
                if args.gait_mode == "prepared_crawl" and not stand and t >= 1.:
                    base_a[:, :2] = (com_reference_acceleration
                        + p[:, 2, None] * (com_reference_xy - whole_com[:, :2])
                        + p[:, 3, None] * (com_reference_velocity - whole_com_velocity[:, :2]))
                base_a[:, 2] = p[:, 2] * (self.height + self.scene.env_origins[:, 2] - pos[:, 2]) - p[:, 3] * body_vel[:, 2]
                # Shortest quaternion error to nominal upright yaw=0.
                rotation_error = -2. * torch.sign(quat[:, :1]) * quat[:, 1:4]
                base_a[:, 3:] = p[:, 4, None] * rotation_error - p[:, 5, None] * angular
                base_a = torch.clamp(base_a, -15., 15.)
                acceleration = torch.cat((base_a, aj), dim=1)
                desired_acceleration = acceleration.clone()
                generalized_velocity = torch.cat((body_vel, angular, self.dq()), dim=1)
                Jstance = Jcontact if args.stance_point != "com" else J
                foot_velocity = torch.einsum("nfij,nj->nfi", Jstance, generalized_velocity)
                if previous_foot_jacobian is None:
                    jacobian_rate_velocity = torch.zeros_like(feet)
                else:
                    jacobian_rate_velocity = torch.einsum("nfij,nj->nfi",
                        (Jstance - previous_foot_jacobian) / self.control_dt, generalized_velocity)
                    # A relatch selects a different material point; do not differentiate across that change.
                    jacobian_rate_velocity = torch.where(refresh_point[..., None], torch.zeros_like(jacobian_rate_velocity), jacobian_rate_velocity)
                previous_foot_jacobian = Jstance.clone()
                stance_error = point_anchor - contact_points if args.stance_point != "com" else contact_anchor - feet
                anchor_acceleration = 100. * stance_error - 20. * foot_velocity - jacobian_rate_velocity
                support_stance = stance & physical_contact if args.gait_mode == "prepared_crawl" and not stand and t >= 1. else stance
                if args.contact_projection and args.controller == "model":
                    acceleration, contact_residual = contact_acceleration_projection(
                        acceleration, Jstance, support_stance, contact_target=anchor_acceleration,
                        weight_matrix=acceleration_weight, regularization=1e-6)
                else:
                    contact_residual = torch.where(support_stance[..., None],
                        torch.einsum("nfij,nj->nfi", Jstance, acceleration) - anchor_acceleration, 0.)
                stance_acceleration_residual = torch.linalg.vector_norm(contact_residual, dim=-1).sum(dim=1) / support_stance.sum(dim=1).clamp_min(1)
                stance_foot_speed = (torch.linalg.vector_norm(foot_velocity, dim=-1) * support_stance).sum(dim=1) / support_stance.sum(dim=1).clamp_min(1)
                sums["stance_acceleration_residual"] += mask_failed_metric(stance_acceleration_residual, alive)
                sums["stance_foot_speed"] += mask_failed_metric(stance_foot_speed, alive)
                dynamics_force = (M @ acceleration[..., None]).squeeze(-1) + h
                A = Jcontact[..., :6].permute(0, 3, 1, 2).reshape(self.n, 6, 12)
                active = support_stance[..., None].expand(-1, -1, 3).reshape(self.n, 12).float()
                A = A * active[:, None, :]
                wanted_wrench = dynamics_force[:, :6]
                if args.force_allocator == "qp":
                    speed_ratio = (self.dq() / 50.).clamp(-1., 1.)
                    velocity_lower = args.effort_fraction * self.limits * torch.maximum(-1. - speed_ratio, -torch.ones_like(speed_ratio))
                    velocity_upper = args.effort_fraction * self.limits * torch.minimum(1. - speed_ratio, torch.ones_like(speed_ratio))
                    tau_lower = torch.maximum(velocity_lower, self.previous - 2500. * self.control_dt)
                    tau_upper = torch.minimum(velocity_upper, self.previous + 2500. * self.control_dt)
                    incompatible = tau_lower > tau_upper
                    emergency = torch.maximum(torch.minimum(self.previous, velocity_upper), velocity_lower)
                    tau_lower = torch.where(incompatible, emergency, tau_lower)
                    tau_upper = torch.where(incompatible, emergency, tau_upper)
                    if force_qp_state is not None:
                        changed = (support_stance != force_qp_previous_stance).any(dim=1) | ~alive
                        force_qp_state = {k: torch.where(changed[:, None], torch.zeros_like(v), v) for k, v in force_qp_state.items()}
                    forces, force_qp_diagnostics, force_qp_state = solve_force_qp(A,
                        Jcontact[..., 6:].permute(0, 3, 1, 2).reshape(self.n, 12, 12),
                        dynamics_force[:, 6:], wanted_wrench, support_stance, limits=self.limits,
                        tau_lower=tau_lower, tau_upper=tau_upper, weight=self.mass * 9.81,
                        iterations=80, warm_start=force_qp_state)
                    force_qp_previous_stance = support_stance.clone()
                    force_qp_feasible = force_qp_diagnostics["feasible"]
                    force_qp_violation = force_qp_diagnostics["max_constraint_violation"]
                    qp_ticks += alive
                    qp_failed_ticks += alive & ~force_qp_feasible
                    qp_max_violation = torch.maximum(qp_max_violation, mask_failed_metric(force_qp_violation, alive))
                    qp_max_torque_violation = torch.maximum(qp_max_torque_violation,
                        mask_failed_metric(force_qp_diagnostics["torque_violation_nm"], alive))
                    qp_max_base_violation = torch.maximum(qp_max_base_violation,
                        mask_failed_metric(force_qp_diagnostics["base_wrench_violation"], alive))
                else:
                    forces = (A.transpose(1, 2) @ torch.linalg.solve(
                        A @ A.transpose(1, 2) + .005 * self.eye6, wanted_wrench[..., None])).squeeze(-1)
                    forces = forces.reshape(self.n, 4, 3) * p[:, 10, None, None]
                    forces[..., 2] = forces[..., 2].clamp(0., 500.)
                    tangential = torch.linalg.vector_norm(forces[..., :2], dim=-1)
                    shrink = torch.minimum(torch.ones_like(tangential), .6 * forces[..., 2] / (tangential + 1e-6))
                    forces[..., :2] *= shrink[..., None]
                    forces *= support_stance[..., None]
                    force_qp_feasible = torch.zeros(self.n, device=self.device, dtype=torch.bool)
                    force_qp_violation = torch.full((self.n,), float("nan"), device=self.device)
                joint_contact = torch.einsum("nfij,nfi->nj", Jcontact[..., 6:], forces)
                residual = wanted_wrench - torch.einsum("nij,nj->ni", A, forces.reshape(self.n, 12))
                sums["base_force_residual"] += mask_failed_metric(
                    torch.linalg.vector_norm(residual[:, :3], dim=1) / self.mass / 9.81, alive)
                sums["base_moment_residual"] += mask_failed_metric(
                    torch.linalg.vector_norm(residual[:, 3:], dim=1) / self.mass / 9.81 / .30, alive)
                sums["model_samples"] += alive
                if args.controller == "pd":
                    raw_requested = 250. * (qref - self.q()) - 8. * self.dq()
                else:
                    raw_requested = dynamics_force[:, 6:] - joint_contact
                requested = torch.clamp(raw_requested, self.previous - 2500. * self.control_dt,
                                         self.previous + 2500. * self.control_dt)
                previous_stance = stance
                previous_presence_contact = physical_contact.clone()
            dq = self.dq()
            state_ok = (torch.isfinite(dq).all(dim=1) & torch.isfinite(self.robot.data.root_state_w).all(dim=1)
                        & (dq.abs().max(dim=1).values < 100.)
                        & (self.robot.data.projected_gravity_b[:, 2] < -.7)
                        & ((self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]) > .23)
                        & ((self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]) < .8))
            alive &= state_ok
            requested = torch.where(alive[:, None], requested, torch.zeros_like(requested))
            before_vel = dq.clone()
            if trace and step % 4 == 0:
                velocity_before = torch.cat((self.robot.data.root_com_lin_vel_w,
                                             self.robot.data.root_com_ang_vel_w, before_vel), dim=1).clone()
            tau = self.advance(requested)
            alive &= (torch.isfinite(self.robot.data.root_state_w).all(dim=1)
                      & torch.isfinite(self.dq()).all(dim=1)
                      & (self.robot.data.projected_gravity_b[:, 2] < -.7)
                      & ((self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]) > .23)
                      & ((self.robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]) < .8)
                      & (self.dq().abs().max(dim=1).values < 100.))
            current_progress, current_speed = forward_metrics(self.robot.data.root_com_pos_w,
                self.robot.data.root_lin_vel_w, initial_xy, axis=self.forward_axis)
            last_safe_progress = torch.where(alive, current_progress, last_safe_progress)
            if t >= 1.:
                measured_steps += 1
                sums["alive_steps"] += alive
                sums["speed"] += torch.where(alive, current_speed, 0.)
                sums["pressure"] += (tau.abs() / self.limits).square().mean(dim=1) * alive
                sums["saturation"] += (tau.abs() / self.limits >= args.effort_fraction * .995).float().mean(dim=1) * alive
                sums["guard_fraction"] += guard_activity(raw_requested, tau).float().mean(dim=1) * alive
                sums["power"] += mask_failed_metric((tau * before_vel).abs().mean(dim=1), alive)
            if trace and step % 4 == 0:
                measured_points_after, ground_normal_after, ground_friction_after = self.contact_details()
                velocity_after = torch.cat((self.robot.data.root_com_lin_vel_w,
                                            self.robot.data.root_com_ang_vel_w, self.dq()), dim=1)
                measured_acceleration = (velocity_after - velocity_before) / DT
                generalized_ground_force = torch.einsum("nfij,nfi->nj", Jcontact,
                                                         ground_normal_after + ground_friction_after)
                actuation = torch.cat((torch.zeros_like(tau[:, :6]), tau), dim=1)
                predicted_acceleration = torch.linalg.solve(M,
                    (actuation + generalized_ground_force - h)[..., None]).squeeze(-1)
                unmodeled_force = (M @ measured_acceleration[..., None]).squeeze(-1) + h - actuation - generalized_ground_force
                data = {"time_s": np.asarray(t + DT), "control_time_s": np.asarray(control_time), "torque_requested_nm": raw_requested.cpu().numpy(),
                        "torque_command_nm": tau.cpu().numpy(), "joint_position_rad": self.q().cpu().numpy(),
                        "joint_velocity_rad_s": self.dq().cpu().numpy(),
                        "root_position_m": (self.robot.data.root_pos_w - self.scene.env_origins).cpu().numpy(),
                        "root_quaternion_wxyz": self.robot.data.root_quat_w.cpu().numpy(), "alive": alive.cpu().numpy(),
                        "contact_force_w": self.scene["feet"].data.net_forces_w.cpu().numpy(),
                        "feet_com_position_m": (self.robot.data.body_com_pos_w[:, self.foot_ids] - self.scene.env_origins[:, None]).cpu().numpy(),
                        "whole_com_position_m": ((self.robot.data.body_com_pos_w * self.body_masses[..., None]).sum(dim=1) / self.mass[:, None] - self.scene.env_origins).cpu().numpy(),
                        "scheduled_stance": stance.cpu().numpy(),
                        "stance_acceleration_residual": stance_acceleration_residual.cpu().numpy(),
                        "stance_foot_speed_m_s": stance_foot_speed.cpu().numpy(),
                        "crawl_stage": crawl_stage.cpu().numpy(), "swing_leg": swing_leg.cpu().numpy(),
                        "measured_support": support_stance.cpu().numpy(),
                        "support_triangle_margin_m": support_margin.cpu().numpy(),
                        "whole_com_target_xy_m": (com_target_xy - self.scene.env_origins[:, :2]).cpu().numpy(),
                        "contact_point_position_m": (contact_points - self.scene.env_origins[:, None]).cpu().numpy(),
                        "allocated_contact_force_w": forces.cpu().numpy(),
                        "desired_base_acceleration": desired_acceleration[:, :6].cpu().numpy(),
                        "projected_base_acceleration": acceleration[:, :6].cpu().numpy(),
                        "base_wrench_residual": residual.cpu().numpy(),
                        "measured_contact_point_position_m": (measured_points_after - self.scene.env_origins[:, None]).cpu().numpy(),
                        "ground_normal_force_w": ground_normal_after.cpu().numpy(), "ground_friction_force_w": ground_friction_after.cpu().numpy(),
                        "root_com_velocity_w": velocity_after[:, :6].cpu().numpy(),
                        "generalized_acceleration_measured_5ms": measured_acceleration.cpu().numpy(),
                        "generalized_acceleration_predicted_from_ground_forces_5ms": predicted_acceleration.cpu().numpy(),
                        "generalized_force_unmodeled_5ms": unmodeled_force.cpu().numpy(),
                        "force_qp_feasible": force_qp_feasible.cpu().numpy(),
                        "force_qp_constraint_violation": force_qp_violation.cpu().numpy(),
                        "whole_com_reference_xy_m": (com_reference_xy - self.scene.env_origins[:, :2]).cpu().numpy(),
                        "whole_com_reference_velocity_m_s": com_reference_velocity.cpu().numpy(),
                        "whole_com_reference_acceleration_m_s2": com_reference_acceleration.cpu().numpy()}
                for key, value in data.items():
                    records[key].append(value.copy())
                if video and step % 8 == 0:
                    video.capture(alive)
        if video:
            video.close()
        duration = max(args.seconds - 1., DT)
        survival = sums["alive_steps"] / max(measured_steps, 1)
        progress = last_safe_progress.cpu().numpy()
        # Fallen robots cannot accrue any benefit after failure.
        summary = {key: (sums[key] / max(measured_steps, 1)).cpu().numpy() for key in
                   ("speed", "pressure", "saturation", "power", "guard_fraction")}
        scores = np.array([candidate_score(float(survival[i]), float(progress[i]),
            float(summary["speed"][i]), float(summary["pressure"][i]), float(summary["guard_fraction"][i]),
            float(summary["power"][i]), 0. if stand else args.speed, duration,
            ended_alive=bool(alive[i])) for i in range(self.n)])
        result = {"seed": seed, "controller": args.controller, "seconds": args.seconds, "speed_command_m_s": 0. if stand else args.speed,
                  "walk_axis_world": args.walk_axis,
                  "contact_projection": args.contact_projection,
                  "gait_mode": args.gait_mode, "liftoff_count": liftoff_count.cpu().tolist(),
                  "stance_point": args.stance_point,
                  "base_acceleration_weight": args.base_acceleration_weight,
                  "smooth_com_reference": args.smooth_com_reference,
                  "joint_friction": args.joint_friction,
                  "joint_dynamic_friction": args.joint_dynamic_friction,
                  "joint_viscous_friction": args.joint_viscous_friction,
                  "force_allocator": args.force_allocator,
                  "qp_control_ticks": qp_ticks.cpu().tolist(),
                  "qp_infeasible_tick_fraction": (qp_failed_ticks / qp_ticks.clamp_min(1)).cpu().tolist(),
                  "qp_max_constraint_violation": qp_max_violation.cpu().tolist(),
                  "qp_max_torque_violation_nm": qp_max_torque_violation.cpu().tolist(),
                  "qp_max_base_wrench_violation": qp_max_base_violation.cpu().tolist(),
                  "completed_step_count": completed_step_count.cpu().tolist(),
                  "survived_envs": int(alive.sum()), "num_envs": self.n,
                  "failed_env_ids": (~alive).nonzero(as_tuple=False).flatten().cpu().tolist(),
                  "failure_handling": "absorbing failure; metric masked and invalid environment quarantined; never revived",
                  "mean_base_force_residual_fraction_of_weight": (sums["base_force_residual"] / sums["model_samples"].clamp_min(1.)).cpu().tolist(),
                  "mean_base_moment_residual_fraction_of_weight_times_0_30m": (sums["base_moment_residual"] / sums["model_samples"].clamp_min(1.)).cpu().tolist(),
                  "mean_stance_acceleration_residual_m_s2": (sums["stance_acceleration_residual"] / sums["model_samples"].clamp_min(1.)).cpu().tolist(),
                  "mean_stance_point_speed_m_s": (sums["stance_foot_speed"] / sums["model_samples"].clamp_min(1.)).cpu().tolist(),
                  "scores": scores.tolist(), "survival_fraction": survival.cpu().tolist(),
                  "progress_m": progress.tolist(), **{k: v.tolist() for k, v in summary.items()}}
        if trace:
            np.savez_compressed(self.out / f"trace_seed{seed}.npz", **{k: np.stack(v) for k, v in records.items()},
                                joint_names=np.array(JOINT_NAMES))
            dump(self.out / f"eval_seed{seed}.json", result)
        return scores, result


def main():
    if args.num_envs < 1 or args.seconds <= 1 or not 1 <= args.control_decimation <= 4:
        raise ValueError("invalid population, duration or digital timing")
    if not 0 < args.effort_fraction < 1:
        raise ValueError("effort fraction requires torque margin")
    if not np.isfinite(args.base_acceleration_weight) or args.base_acceleration_weight <= 0:
        raise ValueError("base acceleration weight must be finite and positive")
    for value in (args.joint_friction, args.joint_dynamic_friction, args.joint_viscous_friction):
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError("joint friction properties must be finite and nonnegative")
    torch.set_num_threads(4)
    plant = Plant()
    if args.mode == "probe":
        rows = np.repeat(DEFAULT[None], plant.n, axis=0)
        scores, report = plant.rollout(rows, args.seed, trace=True, stand=True)
        dump(plant.out / "probe_summary.json", report)
        print(json.dumps({"probe_survived": report["survived_envs"], "n": plant.n,
                          "model_shapes": [18, 18], "jacobian_origin": plant.jacobian_origin}), flush=True)
        if report["survived_envs"] != plant.n:
            raise RuntimeError("V16 standing gate failed; investigate before scaling training")
        return
    if args.mode == "eval":
        value = DEFAULT
        if args.params:
            value = np.array(json.loads(Path(args.params).read_text())["best_parameters"])
        if args.force_allocator == "qp" and not np.isclose(value[10], 1.):
            raise ValueError("QP requires unscaled dynamics wrench, force_scale must be 1")
        _, report = plant.rollout(np.repeat(value[None], plant.n, axis=0), args.seed, trace=True)
        print(json.dumps(report), flush=True)
        return
    if plant.n < 8:
        raise ValueError("CEM training requires population >=8")
    rng = np.random.default_rng(args.seed)
    mean, std = DEFAULT.copy(), .2 * (UPPER - LOWER)
    if args.params:
        mean = np.asarray(json.loads(Path(args.params).read_text())["best_parameters"], dtype=float)
        if mean.shape != DEFAULT.shape or not np.isfinite(mean).all() or np.any(mean < LOWER) or np.any(mean > UPPER):
            raise ValueError("invalid initial CEM parameters")
        if args.force_allocator == "qp" and not np.isclose(mean[10], 1.):
            raise ValueError("QP requires initial force_scale 1")
    best_score, best = -np.inf, mean.copy()
    history = []
    start_generation = 0
    resume_sha = None
    if args.resume:
        restored = resume_cem(json.loads(Path(args.resume).read_text()), seed=args.seed,
                              seconds=args.seconds, control_dt=plant.control_dt, speed=args.speed,
                              effort_fraction=args.effort_fraction, controller=args.controller, walk_axis=args.walk_axis,
                              contact_projection=args.contact_projection, gait_mode=args.gait_mode, stance_point=args.stance_point,
                              joint_friction=args.joint_friction, force_allocator=args.force_allocator,
                              joint_dynamic_friction=args.joint_dynamic_friction, joint_viscous_friction=args.joint_viscous_friction,
                              controller_code_sha256=plant.controller_code_sha256, base_acceleration_weight=args.base_acceleration_weight,
                              smooth_com_reference=args.smooth_com_reference)
        mean, std, best = restored["mean"], restored["std"], restored["best_parameters"]
        best_score, history = restored["best_score"], restored["history"]
        rng, start_generation = restored["rng"], restored["next_generation"]
        resume_sha = sha(args.resume)
    started = time.monotonic()
    for generation in range(start_generation, start_generation + args.generations):
        population = np.clip(rng.normal(mean, std, size=(plant.n, len(mean))), LOWER, UPPER)
        population[0] = mean
        population[1] = best
        if args.force_allocator == "qp":
            population[:, 10] = 1.
        scores, report = plant.rollout(population, args.seed + generation)
        winner = int(np.argmax(scores))
        if scores[winner] > best_score:
            best_score, best = float(scores[winner]), population[winner].copy()
        elite_mean, elite_std = cem_update(population, scores, LOWER, UPPER)
        mean, std = .3 * mean + .7 * elite_mean, .3 * std + .7 * elite_std
        row = {"generation": generation, "score_max": float(scores.max()), "score_mean": float(scores.mean()),
               "survived": report["survived_envs"], "progress_best_m": report["progress_m"][winner],
               "power_best_w_per_joint": report["power"][winner], "seconds_elapsed": time.monotonic() - started}
        row.update(completed_step_count_best=report["completed_step_count"][winner],
                   liftoff_count_best=report["liftoff_count"][winner],
                   winner_ended_alive=winner not in report["failed_env_ids"])
        if args.force_allocator == "qp":
            row.update(qp_infeasible_tick_fraction_best=report["qp_infeasible_tick_fraction"][winner],
                       qp_max_constraint_violation_best=report["qp_max_constraint_violation"][winner],
                       qp_max_torque_violation_nm_best=report["qp_max_torque_violation_nm"][winner])
        history.append(row)
        checkpoint = {"schema": 2, "generation": generation, "best_score": best_score,
                      "best_parameters": best.tolist(), "parameter_names": list(PARAM_NAMES),
                      "mean": mean.tolist(), "std": std.tolist(), "training_seed": args.seed,
                      "rollout_seconds": args.seconds, "control_dt_s": plant.control_dt,
                      "speed_command_m_s": args.speed, "effort_fraction": args.effort_fraction,
                      "controller": args.controller,
                      "walk_axis_world": args.walk_axis,
                      "contact_projection": args.contact_projection,
                      "gait_mode": args.gait_mode,
                      "stance_point": args.stance_point,
                      "base_acceleration_weight": args.base_acceleration_weight,
                      "smooth_com_reference": args.smooth_com_reference,
                      "joint_friction": args.joint_friction,
                      "joint_dynamic_friction": args.joint_dynamic_friction,
                      "joint_viscous_friction": args.joint_viscous_friction,
                      "controller_code_sha256": plant.controller_code_sha256,
                      "force_allocator": args.force_allocator,
                      "rng_state": rng.bit_generator.state, "resumed_checkpoint_sha256": resume_sha,
                      "selection_method": "CEM population training score; requires separate holdout evaluation",
                      "history": history, "manifest_sha256": sha(plant.out / "manifest.json")}
        tmp = plant.out / "checkpoint.tmp"
        dump(tmp, checkpoint)
        tmp.replace(plant.out / "checkpoint.json")
        if generation % 10 == 0 or generation == start_generation + args.generations - 1:
            dump(plant.out / f"checkpoint_{generation:04d}.json", checkpoint)
        print(json.dumps(row), flush=True)
    dump(plant.out / "training_completed.json", {"generations": args.generations, "best_score": best_score,
                                                "holdout_evaluated": False})


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    # NPZ/JSON and any imageio writers have synchronously closed before main returns.
    # SDK 5.1 cleanup stalled after successful job 521; process exit releases the
    # headless job's CUDA resources without waiting for that optional teardown.
    dump(Path(args.output_root) / "run_completed.json", {"mode": args.mode, "artifacts_written": True})
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
