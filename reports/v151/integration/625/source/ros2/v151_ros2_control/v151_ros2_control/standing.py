"""V16-derived all-feet standing WBC using an external MJCF/Pinocchio model.

Joint arrays are FL, FR, RL, RR × HAA, HFE, KFE. Base position is the
MJCF root actor origin in world; base velocities are at that origin in BODY
axes. Quaternions are wxyz. No simulation state other than measured q/v/pose
enters the dynamics. The caller must gate actual support/contact and tilt.
"""
from __future__ import annotations

import numpy as np

from v16_controller_math import digital_acceleration_gains, limit_torque, wbc_acceleration_weights
from v16_physical_model import PhysicalModel, NOMINAL_STAND
from v16_wbc_qp import solve_wbc_qp


def _vector(value, n):
    result = np.asarray(value, dtype=float)
    if result.shape != (n,) or not np.isfinite(result).all():
        raise ValueError(f'Expected {n} finite numbers')
    return result


class StandingController:
    """Standing controller; compute returns (canonical torque, diagnostics).

    First compute latches the measured XY position, yaw, and foot anchors.
    initialize() explicitly relatches them when returning from walking.
    height is an absolute root-origin world Z. Its default is the model's
    nominal standing height, so a crouch hold must pass its measured height.
    Contact points are fixed material points projected from nominal foot COMs
    to z=0 at construction, an approximation requiring physical validation.
    Optional stance_mask selects measured FL/FR/RL/RR support. Its default is
    all four feet. A newly supported foot latches its current material point;
    disabled feet exert zero force and have no contact acceleration constraint.
    Numerical feasibility with fewer contacts is not a stability guarantee.
    forward_offset_m shifts the XY reference along the latched yaw's +y axis.
    It is relative to the initialization position, never accumulated per tick.
    The caller must also provide a compatible joint_reference for that shift.
    """

    def __init__(self, model_path, dt=.02):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive and finite')
        self.dt = float(dt)
        self.model = PhysicalModel(model_path, floating_base=True)
        # PhysX V16 uses .02 rotor inertia, zero viscous damping. Never add the
        # source MJCF's 1.5 Nm/(rad/s) damping to its nonlinear bias.
        self.model.armature_generalized[6:] = .02
        self.nominal_joint_position = NOMINAL_STAND.copy()
        self.nominal_height = float(self.model.keyframes['stand']['base_position'][2])
        self.limits = np.tile([24.8, 48., 48.], 4)
        self.kp, self.kd = digital_acceleration_gains(20.0344, .8295, self.dt)
        self.foot_names = tuple(f'{leg}_foot' for leg in ('FL', 'FR', 'RL', 'RR'))
        records = {record['name']: record for record in self.model.body_records}
        self.contact_local = []
        base = self._base([0., 0., self.nominal_height], [1., 0., 0., 0.])
        for name in self.foot_names:
            local = np.asarray(records[name]['com_body'])
            q = self.model.configuration(self.nominal_joint_position, **base)
            _, pose = self.model._frame_pose(q, name)
            center = pose.translation + pose.rotation @ local
            self.contact_local.append(local + pose.rotation.T @ [0., 0., -center[2]])
        self.contact_local = np.asarray(self.contact_local)
        self.contact_anchors = None
        self._warm = None
        self._previous = np.zeros(12)
        self._previous_reference = None

    @staticmethod
    def _base(position, quaternion):
        quat = _vector(quaternion, 4)
        if np.linalg.norm(quat) < 1e-12:
            raise ValueError('Quaternion must be nonzero')
        return dict(base_position=_vector(position, 3),
                    base_quaternion_xyzw=quat[[1, 2, 3, 0]] / np.linalg.norm(quat))

    def contact_positions(self, joint_pos, base_pos, quat_wxyz):
        base = self._base(base_pos, quat_wxyz)
        return np.array([self.model.frame_position(joint_pos, name, local, **base)
                         for name, local in zip(self.foot_names, self.contact_local)])

    def _jacobians(self, joint_pos, base):
        return np.array([self.model.point_jacobian(joint_pos, name, local, **base)
                         for name, local in zip(self.foot_names, self.contact_local)])

    def initialize(self, joint_pos, base_pos, quat_wxyz, stance_mask=None):
        """Latch actual support anchors; does not change robot state."""
        q = _vector(joint_pos, 12)
        base = self._base(base_pos, quat_wxyz)
        stance = np.ones(4) if stance_mask is None else _vector(stance_mask, 4)
        if not np.all((stance == 0) | (stance == 1)):
            raise ValueError('stance_mask must contain four boolean or binary values')
        self.contact_anchors = self.contact_positions(q, base_pos, quat_wxyz)
        known = self.contact_anchors[stance.astype(bool)] if np.any(stance) else self.contact_anchors
        self._support_plane_z = float(np.median(known[:, 2]))
        self._initial_position = base['base_position'].copy()
        rotation = self.model.pin.Quaternion(base['base_quaternion_xyzw']).matrix()
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        self._target_rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        self._previous_reference = q.copy()
        self._previous_stance = stance.astype(bool)
        self._previous.fill(0.)
        self._warm = None

    def crouch_pose(self, height_drop=.10, forward_shift_m=0.):
        """Return (joint_position12, root_height) preserving nominal contacts.

        This is an initialization pose generator, not a simulated reset or a
        fallen-recovery trajectory. Reject unreachable requested drops.
        Optional forward_shift_m puts the upright root at [0, shift, height]
        while preserving the original nominal foot points. The returned height
        excludes this XY displacement: the caller must apply it separately.
        height_drop=0 returns the corresponding shifted standing pose.
        """
        if not np.isfinite(height_drop) or not 0 <= height_drop <= .18:
            raise ValueError('height_drop must lie in [0, .18] m')
        if not np.isfinite(forward_shift_m):
            raise ValueError('forward_shift_m must be finite')
        q = self.nominal_joint_position.copy()
        height = self.nominal_height - height_drop
        quat = [1., 0., 0., 0.]
        target = self.contact_positions(q, [0., 0., self.nominal_height], quat)
        position = [0., float(forward_shift_m), height]
        base = self._base(position, quat)
        for _ in range(100):
            error = (target - self.contact_positions(q, position, quat)).ravel()
            if np.max(np.abs(error)) < 1e-8:
                return q, height
            jac = self._jacobians(q, base)[:, :, 6:].reshape(12, 12)
            step = np.linalg.solve(jac.T @ jac + 1e-8 * np.eye(12), jac.T @ error)
            q += np.clip(step, -.12, .12)
        raise ValueError('Requested crouch cannot preserve the four contact points')

    def _jacobian_rate_velocity(self, joint_pos, velocity, base):
        # Differentiate with Pinocchio integrate, so changing body axes and
        # canonical/native joint order are included even on the first tick.
        epsilon = 1e-6
        q = self.model.configuration(joint_pos, **base)
        native_v = self.model._to_pin(velocity)
        jacobians = []
        for sign in (-1., 1.):
            advanced = self.model.pin.integrate(self.model.model, q, sign * epsilon * native_v)
            joints = advanced[self.model.joint_q_indices] + self.model.reference_position
            advanced_base = dict(base_position=advanced[:3], base_quaternion_xyzw=advanced[3:7])
            jacobians.append(self._jacobians(joints, advanced_base))
        return np.einsum('fij,j->fi', (jacobians[1] - jacobians[0]) / (2 * epsilon), velocity)

    def contact_consistent_reference(self, joint_pos, base_pos, quat_wxyz, height,
                                     stance_mask, center_body=False, forward_offset_m=0.,
                                     landing_depth_m=.005):
        """Return (qref, world_body_goal_without_forward_offset) after walking.

        center_body is a fraction in [0, 1]: zero uses current XY and one uses
        the four stored contact anchors' centroid and nominal root-to-foot
        centroid offset; intermediate values interpolate these XY goals. Stance
        joints solve IK to their latched material contact points at the desired
        height and latched upright yaw. Swing joints solve at the MEASURED base
        pose toward shifted nominal XY and the support plane minus landing_depth_m.
        The horizontal plane is the median supported anchor Z; without support,
        retain the last plane (entry anchors if no supported sample exists).
        No force or actual touchdown feasibility is inferred from geometric IK.
        A newly measured touchdown updates its anchor before reference solving.
        Call compute with this qref, returned position_reference_world, and the
        SAME forward_offset_m; it is already used by IK but excluded from goal.
        Unreachable stance geometry raises ValueError instead of inventing a
        contact-compatible solution. Reference updates do not move the robot.
        """
        q = _vector(joint_pos, 12)
        base = self._base(base_pos, quat_wxyz)
        stance = _vector(stance_mask, 4)
        if not np.all((stance == 0) | (stance == 1)):
            raise ValueError('stance_mask must contain four boolean or binary values')
        stance = stance.astype(bool)
        if not np.isfinite(height) or not np.isfinite(forward_offset_m):
            raise ValueError('Reference height and forward offset must be finite')
        if not np.isfinite(center_body) or not 0. <= center_body <= 1.:
            raise ValueError('center_body must be a finite fraction in [0, 1]')
        if not np.isfinite(landing_depth_m) or not 0. <= landing_depth_m <= .02:
            raise ValueError('landing_depth_m must be finite and in [0, .02] m')
        if self.contact_anchors is None:
            self.initialize(q, base_pos, quat_wxyz, stance_mask=stance)
        points = self.contact_positions(q, base_pos, quat_wxyz)
        entered = stance & ~self._previous_stance
        self.contact_anchors[entered] = points[entered]
        if np.any(stance):
            self._support_plane_z = float(np.median(self.contact_anchors[stance, 2]))
        goal = base['base_position'].copy()
        if center_body:
            nominal_feet = self.contact_positions(self.nominal_joint_position,
                [0., 0., self.nominal_height], [1., 0., 0., 0.])
            nominal_offset = nominal_feet.mean(axis=0) - [0., 0., self.nominal_height]
            centered = self.contact_anchors.mean(axis=0) - self._target_rotation @ nominal_offset
            goal += float(center_body) * (centered - goal)
        goal[2] = float(height)
        shifted_goal = goal + self._target_rotation @ [0., float(forward_offset_m), 0.]
        target_base = dict(base_position=shifted_goal,
            base_quaternion_xyzw=self.model.pin.Quaternion(self._target_rotation).coeffs())
        reference = self.crouch_pose(0., forward_shift_m=forward_offset_m)[0]
        landing_targets = self.contact_positions(reference, base_pos, quat_wxyz)
        landing_targets[:, 2] = self._support_plane_z - float(landing_depth_m)
        for leg in range(4):
            indices = slice(3 * leg, 3 * leg + 3)
            reference[indices] = q[indices]
            name, local = self.foot_names[leg], self.contact_local[leg]
            leg_base = target_base if stance[leg] else base
            leg_target = self.contact_anchors[leg] if stance[leg] else landing_targets[leg]
            for _ in range(100):
                error = leg_target - self.model.frame_position(reference, name, local, **leg_base)
                if np.max(np.abs(error)) < 1e-7:
                    break
                jac = self.model.point_jacobian(reference, name, local, **leg_base)[:, 6 + 3*leg:9 + 3*leg]
                delta = np.linalg.solve(jac.T @ jac + 1e-8 * np.eye(3), jac.T @ error)
                reference[indices] += np.clip(delta, -.12, .12)
            else:
                kind = 'Stance' if stance[leg] else 'Landing'
                raise ValueError(f'{kind} IK unreachable for {name}: residual {np.max(np.abs(error)):.6g} m')
        return reference, goal

    def compute(self, joint_pos, joint_vel, base_pos, quat_wxyz, lin_vel_b, ang_vel_b,
                *, height=None, joint_reference=None, stance_mask=None, forward_offset_m=0.,
                position_reference_world=None):
        q, dq = _vector(joint_pos, 12), _vector(joint_vel, 12)
        lin, angular = _vector(lin_vel_b, 3), _vector(ang_vel_b, 3)
        base = self._base(base_pos, quat_wxyz)
        target_height = self.nominal_height if height is None else float(height)
        if not np.isfinite(target_height):
            raise ValueError('height must be finite')
        if not np.isfinite(forward_offset_m):
            raise ValueError('forward_offset_m must be finite')
        explicit_position = None if position_reference_world is None else _vector(position_reference_world, 3)
        reference = self.nominal_joint_position if joint_reference is None else _vector(joint_reference, 12)
        stance = np.ones(4) if stance_mask is None else _vector(stance_mask, 4)
        if not np.all((stance == 0) | (stance == 1)):
            raise ValueError('stance_mask must contain four boolean or binary values')
        stance = stance.astype(bool)
        if self.contact_anchors is None:
            self.initialize(q, base_pos, quat_wxyz)
        rotation = self.model.pin.Quaternion(base['base_quaternion_xyzw']).matrix()
        velocity = np.r_[lin, angular, dq]
        mass = self.model.mass_matrix(q, **base)
        bias = self.model.bias(q, velocity, include_damping=False, **base)
        jac = self._jacobians(q, base)
        points = self.contact_positions(q, base_pos, quat_wxyz)
        entered = stance & ~self._previous_stance
        self.contact_anchors[entered] = points[entered]
        if np.any(stance != self._previous_stance):
            # Force variables and contact rows change when support changes.
            self._warm = None
        jdot_v = self._jacobian_rate_velocity(q, velocity, base)
        target = 100. * (self.contact_anchors - points) - 20. * np.einsum('fij,j->fi', jac, velocity) - jdot_v
        position_reference = self._initial_position.copy() if explicit_position is None else explicit_position.copy()
        if explicit_position is None:
            position_reference[2] = target_height
        position_reference += self._target_rotation @ [0., float(forward_offset_m), 0.]
        position_error = position_reference - base['base_position']
        acceleration_world = np.clip(33.0577 * position_error - 19.5438 * (rotation @ lin), -15., 15.)
        # Pin free-flyer acceleration is d(v_body)/dt, hence the transport term.
        base_linear = rotation.T @ acceleration_world - np.cross(angular, lin)
        rotation_error = self.model.pin.log3(rotation.T @ self._target_rotation)
        base_angular = np.clip(32.1831 * rotation_error - 12.6411 * angular, -15., 15.)
        reference_velocity = np.clip((reference - self._previous_reference) / self.dt, -3., 3.)
        joint_acceleration = np.clip(self.kp * (1. + .6475) * (reference - q)
                                     + self.kd * (reference_velocity - dq), -150., 150.)
        desired = np.r_[base_linear, base_angular, joint_acceleration]
        speed = np.clip(dq / 50., -1., 1.)
        velocity_lower = .9 * self.limits * np.maximum(-1. - speed, -1.)
        velocity_upper = .9 * self.limits * np.minimum(1. - speed, 1.)
        lower = np.maximum(velocity_lower, self._previous - 2500. * self.dt)
        upper = np.minimum(velocity_upper, self._previous + 2500. * self.dt)
        incompatible = lower > upper
        emergency = np.clip(self._previous, velocity_lower, velocity_upper)
        lower, upper = np.where(incompatible, emergency, lower), np.where(incompatible, emergency, upper)
        acceleration, forces, requested, diagnostics, warm = solve_wbc_qp(
            mass, bias, jac, desired, stance, contact_target=target,
            limits=self.limits, tau_lower=lower, tau_upper=upper,
            weight=self.model.total_mass * 9.81,
            acceleration_weights=wbc_acceleration_weights(1000., 'unscaled'),
            iterations=320, warm_start=self._warm)
        if not bool(diagnostics['finite']):
            raise RuntimeError('Standing WBC produced non-finite output')
        torque = limit_torque(requested[None], dq[None], self.limits,
                              self._previous[None], dt=self.dt)[0]
        self._previous, self._previous_reference, self._warm = torque.copy(), reference.copy(), warm
        self._previous_stance = stance.copy()
        diagnostics.update(acceleration=acceleration, forces_world=forces,
                           base_position_reference_world=position_reference.copy(),
                           requested_torque=requested, commanded_torque=torque.copy(),
                           guard_delta_nm=float(np.max(np.abs(torque - requested))),
                           anchor_error_m=float(np.max(np.abs(np.where(stance[:, None], self.contact_anchors - points, 0.)))),
                           support_count=int(np.count_nonzero(stance)), stance_mask=stance.copy(),
                           model_source_sha256=self.model.source_sha256,
                           contact_assumption='all_four_supported' if stance_mask is None else 'measured_stance_mask',
                           bias_includes_damping=False)
        return torque, diagnostics
