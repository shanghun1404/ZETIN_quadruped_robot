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
    """All-four-support controller; compute returns (canonical torque, diagnostics).

    First compute latches the measured XY position, yaw, and foot anchors.
    initialize() explicitly relatches them when returning from walking.
    height is an absolute root-origin world Z. Its default is the model's
    nominal standing height, so a crouch hold must pass its measured height.
    Contact points are fixed material points projected from nominal foot COMs
    to z=0 at construction, an approximation requiring physical validation.
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

    def initialize(self, joint_pos, base_pos, quat_wxyz):
        """Latch actual support anchors; does not change robot state."""
        q = _vector(joint_pos, 12)
        base = self._base(base_pos, quat_wxyz)
        self.contact_anchors = self.contact_positions(q, base_pos, quat_wxyz)
        self._initial_position = base['base_position'].copy()
        rotation = self.model.pin.Quaternion(base['base_quaternion_xyzw']).matrix()
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        self._target_rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        self._previous_reference = q.copy()
        self._previous.fill(0.)
        self._warm = None

    def crouch_pose(self, height_drop=.10):
        """Return (joint_position12, root_height) preserving nominal contacts.

        This is an initialization pose generator, not a simulated reset or a
        fallen-recovery trajectory. Reject unreachable requested drops.
        """
        if not np.isfinite(height_drop) or not 0 <= height_drop <= .18:
            raise ValueError('height_drop must lie in [0, .18] m')
        q = self.nominal_joint_position.copy()
        height = self.nominal_height - height_drop
        quat = [1., 0., 0., 0.]
        target = self.contact_positions(q, [0., 0., self.nominal_height], quat)
        base = self._base([0., 0., height], quat)
        for _ in range(100):
            error = (target - self.contact_positions(q, [0., 0., height], quat)).ravel()
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

    def compute(self, joint_pos, joint_vel, base_pos, quat_wxyz, lin_vel_b, ang_vel_b,
                *, height=None, joint_reference=None):
        q, dq = _vector(joint_pos, 12), _vector(joint_vel, 12)
        lin, angular = _vector(lin_vel_b, 3), _vector(ang_vel_b, 3)
        base = self._base(base_pos, quat_wxyz)
        target_height = self.nominal_height if height is None else float(height)
        if not np.isfinite(target_height):
            raise ValueError('height must be finite')
        reference = self.nominal_joint_position if joint_reference is None else _vector(joint_reference, 12)
        if self.contact_anchors is None:
            self.initialize(q, base_pos, quat_wxyz)
        rotation = self.model.pin.Quaternion(base['base_quaternion_xyzw']).matrix()
        velocity = np.r_[lin, angular, dq]
        mass = self.model.mass_matrix(q, **base)
        bias = self.model.bias(q, velocity, include_damping=False, **base)
        jac = self._jacobians(q, base)
        points = self.contact_positions(q, base_pos, quat_wxyz)
        jdot_v = self._jacobian_rate_velocity(q, velocity, base)
        target = 100. * (self.contact_anchors - points) - 20. * np.einsum('fij,j->fi', jac, velocity) - jdot_v
        position_error = self._initial_position - base['base_position']
        position_error[2] = target_height - base['base_position'][2]
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
            mass, bias, jac, desired, np.ones(4), contact_target=target,
            limits=self.limits, tau_lower=lower, tau_upper=upper,
            weight=self.model.total_mass * 9.81,
            acceleration_weights=wbc_acceleration_weights(1000., 'unscaled'),
            iterations=320, warm_start=self._warm)
        if not bool(diagnostics['finite']):
            raise RuntimeError('Standing WBC produced non-finite output')
        torque = limit_torque(requested[None], dq[None], self.limits,
                              self._previous[None], dt=self.dt)[0]
        self._previous, self._previous_reference, self._warm = torque.copy(), reference.copy(), warm
        diagnostics.update(acceleration=acceleration, forces_world=forces,
                           requested_torque=requested, commanded_torque=torque.copy(),
                           guard_delta_nm=float(np.max(np.abs(torque - requested))),
                           anchor_error_m=float(np.max(np.abs(self.contact_anchors - points))),
                           model_source_sha256=self.model.source_sha256,
                           contact_assumption='all_four_supported', bias_includes_damping=False)
        return torque, diagnostics
