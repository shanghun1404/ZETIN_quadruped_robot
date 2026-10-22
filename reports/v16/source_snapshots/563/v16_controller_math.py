"""Digital feedback, torque envelope and CEM selection, without simulator imports."""
from __future__ import annotations
import math
import numpy as np

JOINT_NAMES = tuple(f"{leg}_{joint}" for leg in ("FL", "FR", "RL", "RR")
                    for joint in ("HAA", "HFE", "KFE"))
PARAM_NAMES = ("joint_frequency_rad_s", "joint_damping_ratio", "body_position_gain",
               "body_velocity_gain", "body_attitude_gain", "body_angular_gain",
               "period_s", "step_length_m", "step_height_m", "duty_factor",
               "force_scale", "posture_gain")
LOWER = np.array([6., .7, 20., 3., 20., 3., 3., .04, .015, .80, .7, .1])
UPPER = np.array([22., 1.4, 160., 20., 160., 20., 6., .20, .04, .90, 1.1, .8])
DEFAULT = np.array([12., 1., 70., 10., 70., 10., 4., .10, .020, .82, 1., .25])


def rigid_point_jacobian(linear, angular, displacement_world):
    """Translate a rigid COM Jacobian to the SAME physical point used for force.

    J_point = J_COM - skew(r_world) J_angular. r is a rigid local displacement
    rotated into world axes, not a point that is reprojected every time step.
    """
    r = displacement_world
    rows = (linear[..., 0, :] + r[..., 2, None] * angular[..., 1, :] - r[..., 1, None] * angular[..., 2, :],
            linear[..., 1, :] + r[..., 0, None] * angular[..., 2, :] - r[..., 2, None] * angular[..., 0, :],
            linear[..., 2, :] + r[..., 1, None] * angular[..., 0, :] - r[..., 0, None] * angular[..., 1, :])
    if hasattr(linear, "new_tensor"):
        import torch
        return torch.stack(rows, dim=-2)
    return np.stack(rows, axis=-2)


def add_joint_armature(rigid_mass, armature):
    """Add measured rotor inertia to the joint block of a floating-base matrix.

    The raw PhysX tensor is retained separately; HPC comparison 526 showed it
    matches source rigid inertia rather than the armature-inclusive inertia.
    """
    result = rigid_mass.clone() if hasattr(rigid_mass, "clone") else rigid_mass.copy()
    ids = list(range(6, result.shape[-1]))
    result[..., ids, ids] += armature
    return result


def forward_metrics(position, velocity, initial_xy, *, axis):
    """Measure along the commanded axis; sideways travel provides no progress."""
    return position[..., axis] - initial_xy[..., axis], velocity[..., axis]


def joint_permutation(actual_names):
    names = list(actual_names)
    if len(names) != len(JOINT_NAMES) or len(set(names)) != len(names) or set(names) != set(JOINT_NAMES):
        raise ValueError("plant must expose each named V16 joint exactly once")
    ids = [names.index(name) for name in JOINT_NAMES]
    return ids, list(range(6)) + [index + 6 for index in ids]


def digital_acceleration_gains(frequency: float, damping: float, dt: float) -> tuple[float, float]:
    """Place the exact-ZOH unit-inertia double-integrator closed-loop poles.

    Torque is held over dt; this is not a continuous PD gain silently reused.
    """
    if not all(math.isfinite(v) and v > 0 for v in (frequency, damping, dt)):
        raise ValueError("frequency, damping and sample interval must be positive and finite")
    if damping <= 1.:
        total = 2 * math.exp(-damping * frequency * dt) * math.cos(
            frequency * math.sqrt(1 - damping * damping) * dt)
    else:
        delta = math.sqrt(damping * damping - 1)
        total = math.exp((-damping + delta) * frequency * dt) + math.exp(
            (-damping - delta) * frequency * dt)
    product = math.exp(-2 * damping * frequency * dt)
    kp = (1 - total + product) / (dt * dt)
    kd = (1 + .5 * dt * dt * kp - product) / dt
    return kp, kd


def limit_torque(requested, velocity, limits, previous, *, dt, fraction=.9,
                 slew_nm_s=2500., velocity_limit=50.):
    requested, velocity, limits, previous = [np.asarray(v) for v in
                                            (requested, velocity, limits, previous)]
    if (requested.ndim != 2 or requested.shape != velocity.shape or previous.shape != requested.shape
            or limits.shape != (requested.shape[-1],) or np.any(limits <= 0)
            or not all(np.isfinite(v).all() for v in (requested, velocity, limits, previous))
            or not all(math.isfinite(v) for v in (dt, fraction, slew_nm_s, velocity_limit))
            or dt <= 0 or not 0 < fraction <= 1 or slew_nm_s <= 0 or velocity_limit <= 0):
        raise ValueError("invalid torque guard input")
    speed = np.clip(velocity / velocity_limit, -1, 1)
    upper = fraction * limits * np.minimum(1 - speed, 1)
    lower = fraction * limits * np.maximum(-1 - speed, -1)
    slewed = np.clip(requested, previous - slew_nm_s * dt, previous + slew_nm_s * dt)
    return np.clip(slewed, lower, upper)


def candidate_score(survival, progress, speed, effort_pressure, saturation, power,
                    target_speed, seconds, *, ended_alive=True):
    """Survival is the first gate, progress the second; stopping cannot win on effort."""
    values = [survival, progress, speed, effort_pressure, saturation, power, target_speed, seconds]
    if not np.isfinite(values).all() or seconds <= 0 or target_speed < 0:
        return -1e9
    if not ended_alive or survival < .99:
        return -10000. + 1000. * survival + 20. * min(progress, target_speed * seconds)
    if target_speed > 0 and progress < .25 * target_speed * seconds:
        return -1000. + 100. * max(progress, 0.)
    return (100 * min(progress / max(target_speed * seconds, .1), 1.)
            - 30 * abs(speed - target_speed) - 8 * effort_pressure
            - 10 * saturation - .025 * power)


def swing_reference(start, end, phase, height):
    """Array-generic minimum-jerk interpolation with zero endpoint velocity."""
    blend = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
    result = start + blend[..., None] * (end - start)
    result[..., 2] += 16 * height * phase**2 * (1 - phase)**2
    return result


def guard_activity(requested, command):
    """Record any request-to-command projection, including envelope and slew."""
    return abs(requested - command) > 1e-4


def mask_failed_metric(values, alive):
    """Boolean assignment, unlike NaN*0, preserves valid peers and failed evidence."""
    result = values.clone() if hasattr(values, "clone") else values.copy()
    result[~alive] = 0.
    return result


def resume_cem(state, *, seed, seconds, control_dt, speed=.25, effort_fraction=.90,
               controller="model", walk_axis="y", contact_projection=False, gait_mode="periodic", stance_point="com",
               joint_friction=.3, force_allocator="ls", joint_dynamic_friction=None, joint_viscous_friction=None,
               controller_code_sha256=None, base_acceleration_weight=10., smooth_com_reference=False):
    """Restore a random stream and distribution only for an unchanged objective."""
    if (state.get("schema") != 2 or state.get("parameter_names") != list(PARAM_NAMES)
            or state.get("training_seed") != seed or state.get("rollout_seconds") != seconds
            or state.get("control_dt_s") != control_dt or state.get("speed_command_m_s") != speed
            or state.get("effort_fraction") != effort_fraction or state.get("controller") != controller
            or state.get("walk_axis_world") != walk_axis
            or state.get("contact_projection", False) != contact_projection
            or state.get("gait_mode", "periodic") != gait_mode
            or state.get("stance_point", "com") != stance_point
            or state.get("joint_friction", .3) != joint_friction
            or state.get("force_allocator", "ls") != force_allocator
            or state.get("joint_dynamic_friction") != joint_dynamic_friction
            or state.get("joint_viscous_friction") != joint_viscous_friction
            or state.get("controller_code_sha256") != controller_code_sha256
            or state.get("base_acceleration_weight", 10.) != base_acceleration_weight
            or state.get("smooth_com_reference", False) != smooth_com_reference):
        raise ValueError("checkpoint schema, parameter order, seed or rollout objective changed")
    result = {k: np.asarray(state[k], dtype=float) for k in ("mean", "std", "best_parameters")}
    if (any(v.shape != LOWER.shape or not np.isfinite(v).all() for v in result.values())
            or np.any(result["std"] <= 0) or not np.isfinite(state["best_score"])
            or np.any(result["mean"] < LOWER) or np.any(result["mean"] > UPPER)
            or np.any(result["best_parameters"] < LOWER) or np.any(result["best_parameters"] > UPPER)):
        raise ValueError("invalid CEM checkpoint arrays")
    generation = state.get("generation")
    if not isinstance(generation, int) or generation < 0:
        raise ValueError("invalid generation")
    rng = np.random.default_rng(seed)
    rng.bit_generator.state = state["rng_state"]
    result.update(rng=rng, next_generation=generation + 1,
                  best_score=float(state["best_score"]), history=list(state["history"]))
    return result


def cem_update(samples, scores, lower, upper, *, elite_fraction=.2):
    samples, scores, lower, upper = [np.asarray(v, dtype=float) for v in
                                    (samples, scores, lower, upper)]
    if (samples.ndim != 2 or scores.shape != (samples.shape[0],) or samples.shape[0] < 2
            or lower.shape != (samples.shape[1],) or upper.shape != lower.shape
            or np.any(upper <= lower) or not 0 < elite_fraction <= 1
            or not all(np.isfinite(v).all() for v in (samples, scores, lower, upper))):
        raise ValueError("invalid CEM population or bounds")
    count = max(2, math.ceil(len(samples) * elite_fraction))
    elite = samples[np.argsort(scores)[-count:]]
    return np.clip(elite.mean(axis=0), lower, upper), np.maximum(elite.std(axis=0), .01 * (upper - lower))
