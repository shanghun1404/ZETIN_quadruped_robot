"""Named MJCF rigid-body dynamics for V16, using Pinocchio 2.7.

This parser deliberately covers the explicit-inertial, local-coordinate MJCF
used by robot_v2. It rejects unsupported physics rather than guessing. Meshes
are not needed for dynamics; body frames are not assumed to be contact points.
All public joint arrays use JOINT_NAMES; free-base vectors use base6,joints12.
Pinocchio base twist convention is body-local [linear, angular] in SI units.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

JOINT_NAMES = tuple(f"{leg}_{joint}" for leg in ("FL", "FR", "RL", "RR") for joint in ("HAA", "HFE", "KFE"))
NOMINAL_STAND = np.tile([0.0, -0.55, 1.10], 4)


def _vec(value, size, default=None):
    result = np.asarray(default if value is None else value.split(), dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected {size} finite numbers: {value!r}")
    return result


def _axis_rotation(axis, theta):
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-14:
        raise ValueError("Zero rotation axis")
    x, y, z = axis / norm
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * (skew @ skew)


def mjcf_rotation(element, *, angle="degree", eulerseq="xyz"):
    """MJCF lower-case axes are intrinsic; upper-case axes are extrinsic.

    Primary source: MuJoCo XML reference compiler/eulerseq. In particular,
    MJCF xyz = Rx @ Ry @ Rz, whereas URDF rpy corresponds to MJCF XYZ.
    """
    attrs = [key for key in ("quat", "euler", "axisangle", "xyaxes", "zaxis") if key in element.attrib]
    if len(attrs) > 1:
        raise ValueError("Multiple orientation representations")
    if not attrs:
        return np.eye(3)
    if attrs[0] in ("xyaxes", "zaxis"):
        raise ValueError("xyaxes/zaxis orientation not supported by this source-specific parser")
    if attrs[0] == "quat":
        w, x, y, z = _vec(element.get("quat"), 4)
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm < 1e-14:
            raise ValueError("Zero quaternion")
        w, x, y, z = np.array([w, x, y, z]) / norm
        return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    factor = np.pi / 180 if angle == "degree" else 1.0
    if angle not in ("degree", "radian"):
        raise ValueError(f"Unknown angle convention: {angle}")
    if attrs[0] == "axisangle":
        value = _vec(element.get("axisangle"), 4)
        return _axis_rotation(value[:3], value[3] * factor)
    if len(eulerseq) != 3 or any(axis not in "xyzXYZ" for axis in eulerseq):
        raise ValueError("Invalid Euler sequence")
    rotation = np.eye(3)
    for axis, theta in zip(eulerseq, _vec(element.get("euler"), 3) * factor):
        elementary = _axis_rotation(np.eye(3)["xyz".index(axis.lower())], theta)
        rotation = rotation @ elementary if axis.islower() else elementary @ rotation
    return rotation


class PhysicalModel:
    """One Pinocchio model and its explicit source-to-canonical mappings."""

    def __init__(self, mjcf_path, floating_base=False, joint_names=JOINT_NAMES):
        import pinocchio as pin
        self.pin = pin
        self.source_path = str(Path(mjcf_path).resolve())
        self.source_sha256 = hashlib.sha256(Path(mjcf_path).read_bytes()).hexdigest()
        root = ET.parse(mjcf_path).getroot()
        if root.find("include") is not None:
            raise ValueError("Expand MJCF includes before parsing")
        compiler = root.find("compiler")
        options = {} if compiler is None else compiler.attrib
        self.angle = options.get("angle", "degree")
        self.eulerseq = options.get("eulerseq", "xyz")
        if options.get("coordinate", "local") != "local" or options.get("inertiafromgeom") == "true":
            raise ValueError("Only local explicit-inertial MJCF is supported")
        if float(options.get("settotalmass", "-1")) > 0 or options.get("balanceinertia", "false") == "true":
            raise ValueError("Compiler inertia rescaling is unsupported")
        self.floating_base = bool(floating_base)
        self.joint_names = tuple(joint_names)
        self.model = pin.Model()
        option = root.find("option")
        self.model.gravity.linear = _vec(None if option is None else option.get("gravity"), 3, [0, 0, -9.81])
        self.model.gravity.angular = np.zeros(3)
        defaults = root.find("default")
        if defaults is not None and defaults.find("default") is not None:
            raise ValueError("Nested MJCF default classes are unsupported")
        self._joint_defaults = {} if defaults is None or defaults.find("joint") is None else dict(defaults.find("joint").attrib)
        self._joint_records = {}
        self.body_names = []
        self.body_records = []
        self.source_total_mass = 0.0
        self._traversal_joints = []
        self.default_base_position = np.zeros(3)
        self.default_base_quaternion_xyzw = np.array([0., 0., 0., 1.])
        world = root.find("worldbody")
        if world is None or len(world.findall("body")) != 1:
            raise ValueError("Expected one robot root body")

        def placement(element):
            return pin.SE3(mjcf_rotation(element, angle=self.angle, eulerseq=self.eulerseq), _vec(element.get("pos"), 3, [0, 0, 0]))

        def visit(body, parent_joint, parent_to_body, parent_frame):
            if body.get("childclass") or body.get("class"):
                raise ValueError("MJCF classes are unsupported")
            name = body.get("name")
            if not name or name in self.body_names:
                raise ValueError("Body names must be present and unique")
            self.body_names.append(name)
            transform = parent_to_body * placement(body)
            joints = body.findall("joint") + body.findall("freejoint")
            if len(joints) > 1:
                raise ValueError("Multiple joints on one MJCF body are unsupported")
            joint_id = parent_joint
            if joints:
                joint = joints[0]
                kind = "free" if joint.tag == "freejoint" else joint.get("type", "hinge")
                if kind == "free":
                    if parent_joint != 0 or name != world.find("body").get("name"):
                        raise ValueError("Free joint must be on the robot root")
                    self.default_base_position = transform.translation.copy()
                    self.default_base_quaternion_xyzw = pin.Quaternion(transform.rotation).coeffs().copy()
                    if floating_base:
                        joint_id = self.model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), joint.get("name", "root"))
                        transform = pin.SE3.Identity()
                elif kind == "hinge":
                    props = dict(self._joint_defaults)
                    props.update(joint.attrib)
                    if props.get("class") or float(props.get("stiffness", "0")) or float(props.get("frictionloss", "0")):
                        raise ValueError("Classes, springs and dry friction are unsupported")
                    if "springdamper" in props:
                        raise ValueError("Automatic springdamper inference is unsupported")
                    axis = _vec(props.get("axis"), 3, [0, 0, 1])
                    if np.linalg.norm(axis) < 1e-14:
                        raise ValueError("Zero joint axis")
                    axis /= np.linalg.norm(axis)
                    anchor = _vec(props.get("pos"), 3, [0, 0, 0])
                    jname = props["name"]
                    if jname in self._joint_records:
                        raise ValueError("Duplicate joint name")
                    joint_id = self.model.addJoint(parent_joint, pin.JointModelRevoluteUnaligned(axis), transform * pin.SE3(np.eye(3), anchor), jname)
                    transform = pin.SE3(np.eye(3), -anchor)
                    ref = float(props.get("ref", "0")) * (np.pi/180 if self.angle == "degree" else 1)
                    self._joint_records[jname] = dict(id=joint_id, damping=float(props.get("damping", "0")), armature=float(props.get("armature", "0")), reference=ref, axis_body=axis.tolist(), anchor_body=anchor.tolist(), body=name)
                    self._traversal_joints.append(jname)
                else:
                    raise ValueError(f"Unsupported joint type: {kind}")
            inertial = body.find("inertial")
            if inertial is None:
                raise ValueError(f"Missing explicit inertia: {name}")
            mass = float(inertial.attrib["mass"])
            com = _vec(inertial.get("pos"), 3)
            if "fullinertia" in inertial.attrib:
                if any(attr in inertial.attrib for attr in ("quat", "euler", "axisangle", "diaginertia")):
                    raise ValueError("fullinertia with redundant inertia orientation")
                xx, yy, zz, xy, xz, yz = _vec(inertial.get("fullinertia"), 6)
                inertia = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            else:
                rotation = mjcf_rotation(inertial, angle=self.angle, eulerseq=self.eulerseq)
                inertia = rotation @ np.diag(_vec(inertial.get("diaginertia"), 3)) @ rotation.T
            if mass <= 0 or np.linalg.eigvalsh(inertia).min() <= 0:
                raise ValueError(f"Nonphysical mass/inertia: {name}")
            self.model.appendBodyToJoint(joint_id, pin.Inertia(mass, com, inertia), transform)
            frame_id = self.model.addFrame(pin.Frame(name, joint_id, parent_frame, transform, pin.FrameType.BODY))
            self.source_total_mass += mass
            self.body_records.append(dict(name=name, mass=mass, com_body=com.tolist(), inertia_com_body=inertia.tolist(), parent_pin_joint=int(joint_id), joint_to_body_translation=transform.translation.tolist(), joint_to_body_rotation=transform.rotation.tolist()))
            for child in body.findall("body"):
                visit(child, joint_id, transform, frame_id)

        visit(world.find("body"), 0, pin.SE3.Identity(), 0)
        if set(self._joint_records) != set(self.joint_names) or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("Source joints must exactly match the requested named order")
        records = [self._joint_records[name] for name in self.joint_names]
        self.joint_q_indices = np.array([self.model.joints[r["id"]].idx_q for r in records], dtype=int)
        self.joint_v_indices = np.array([self.model.joints[r["id"]].idx_v for r in records], dtype=int)
        self.canonical_v_indices = np.r_[np.arange(6), self.joint_v_indices] if floating_base else self.joint_v_indices.copy()
        self.source_damping = np.array([r["damping"] for r in records])
        self.source_armature = np.array([r["armature"] for r in records])
        self.reference_position = np.array([r["reference"] for r in records])
        self.damping_generalized = np.r_[np.zeros(6), self.source_damping] if floating_base else self.source_damping.copy()
        self.armature_generalized = np.r_[np.zeros(6), self.source_armature] if floating_base else self.source_armature.copy()
        # Keep Pinocchio's internal armature zero: its algorithms differ by version.
        # Add source armature explicitly in CRBA/RNEA and solve that physical M.
        self.model.armature = np.zeros(self.model.nv)
        self.actuation_matrix = np.zeros((len(records), self.model.nv))
        self.actuation_matrix[:, (6 if floating_base else 0):] = np.eye(len(records))
        self.body_count = len(self.body_names)
        self.total_mass = sum(float(inertia.mass) for inertia in self.model.inertias)
        motors = {}
        actuator = root.find("actuator")
        if actuator is not None:
            for motor in actuator:
                if motor.tag != "motor" or motor.get("joint") not in self._joint_records:
                    raise ValueError("Only joint motor actuators are supported")
                gear = _vec(motor.get("gear"), 6, [1, 0, 0, 0, 0, 0])
                if not np.array_equal(gear, [1, 0, 0, 0, 0, 0]):
                    raise ValueError("Nonunit motor gear is unsupported")
                limits = _vec(motor.get("ctrlrange"), 2)
                motors[motor.get("joint")] = float(np.max(np.abs(limits)))
        self.effort_limits = np.array([motors[name] for name in self.joint_names])
        self.keyframes = {}
        for key in root.findall("keyframe/key"):
            qpos = _vec(key.get("qpos"), len(records) + 7)
            traversal_values = dict(zip(self._traversal_joints, qpos[7:]))
            self.keyframes[key.get("name")] = dict(base_position=qpos[:3].tolist(), base_quaternion_xyzw=qpos[[4, 5, 6, 3]].tolist(), joint_position=[traversal_values[name] for name in self.joint_names])
        self.data = self.model.createData()

    def configuration(self, joint_position, *, base_position=None, base_quaternion_xyzw=None):
        q = self.pin.neutral(self.model)
        joints = _vec_array(joint_position, len(self.joint_names))
        q[self.joint_q_indices] = joints - self.reference_position
        if self.floating_base:
            q[:3] = self.default_base_position if base_position is None else _vec_array(base_position, 3)
            quat = self.default_base_quaternion_xyzw if base_quaternion_xyzw is None else _vec_array(base_quaternion_xyzw, 4)
            if np.linalg.norm(quat) < 1e-14:
                raise ValueError("Zero base quaternion")
            q[3:7] = quat / np.linalg.norm(quat)
        elif base_position is not None or base_quaternion_xyzw is not None:
            raise ValueError("Fixed base placement is fixed to source MJCF")
        return q

    def velocity(self, joint_velocity, *, base_twist=None):
        """Construct a canonical velocity accepted by every public dynamics API."""
        joints = _vec_array(joint_velocity, len(self.joint_names))
        canonical = np.r_[np.zeros(6) if base_twist is None else _vec_array(base_twist, 6), joints] if self.floating_base else joints
        return canonical

    def _to_pin(self, canonical):
        result = np.empty(self.model.nv)
        result[self.canonical_v_indices] = _vec_array(canonical, self.model.nv)
        return result

    def _from_pin(self, native):
        return np.asarray(native)[self.canonical_v_indices].copy()

    def mass_matrix(self, joint_position, *, include_armature=True, **base):
        matrix = np.asarray(self.pin.crba(self.model, self.data, self.configuration(joint_position, **base))).copy()
        matrix = (matrix + matrix.T) / 2
        matrix = matrix[np.ix_(self.canonical_v_indices, self.canonical_v_indices)]
        return matrix + np.diag(self.armature_generalized) if include_armature else matrix

    def gravity(self, joint_position, **base):
        return self._from_pin(self.pin.computeGeneralizedGravity(self.model, self.data, self.configuration(joint_position, **base)))

    def bias(self, joint_position, velocity, *, include_damping=True, **base):
        velocity = _vec_array(velocity, self.model.nv)
        result = self._from_pin(self.pin.nonLinearEffects(self.model, self.data, self.configuration(joint_position, **base), self._to_pin(velocity)))
        return result + self.damping_generalized * velocity if include_damping else result

    def inverse_dynamics(self, joint_position, velocity, acceleration, *, include_armature=True, include_damping=True, **base):
        velocity = _vec_array(velocity, self.model.nv)
        acceleration = _vec_array(acceleration, self.model.nv)
        result = self._from_pin(self.pin.rnea(self.model, self.data, self.configuration(joint_position, **base), self._to_pin(velocity), self._to_pin(acceleration)))
        if include_armature:
            result += self.armature_generalized * acceleration
        if include_damping:
            result += self.damping_generalized * velocity
        return result

    def forward_dynamics(self, joint_position, velocity, generalized_effort, *, include_armature=True, include_damping=True, **base):
        velocity = _vec_array(velocity, self.model.nv)
        effort = _vec_array(generalized_effort, self.model.nv)
        if include_armature and np.any(self.armature_generalized):
            return np.linalg.solve(self.mass_matrix(joint_position, **base), effort - self.bias(joint_position, velocity, include_damping=include_damping, **base))
        if include_damping:
            effort = effort - self.damping_generalized * velocity
        return self._from_pin(self.pin.aba(self.model, self.data, self.configuration(joint_position, **base), self._to_pin(velocity), self._to_pin(effort)))

    def _frame_pose(self, q, body_name):
        frame = self.model.getFrameId(body_name)
        if frame >= len(self.model.frames):
            raise ValueError(f"Unknown body frame: {body_name}")
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        return frame, self.data.oMf[frame]

    def frame_position(self, joint_position, body_name, point_local=(0, 0, 0), **base):
        _, pose = self._frame_pose(self.configuration(joint_position, **base), body_name)
        return pose.translation + pose.rotation @ _vec_array(point_local, 3)

    def point_jacobian(self, joint_position, body_name, point_local=(0, 0, 0), **base):
        q = self.configuration(joint_position, **base)
        frame, pose = self._frame_pose(q, body_name)
        jac = np.asarray(self.pin.computeFrameJacobian(self.model, self.data, q, frame, self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)).reshape(6, self.model.nv)
        offset = pose.rotation @ _vec_array(point_local, 3)
        point_jac = jac[:3] + np.cross(jac[3:].T, offset).T
        return point_jac[:, self.canonical_v_indices].copy()

    def points_kinematics(self, joint_position, body_names, points_local, *, with_jacobians=False, **base):
        """World material points and optional canonical Jacobians from one FK.

        This performs the same LOCAL_WORLD_ALIGNED point shift as
        point_jacobian, sharing configuration and tree traversal across frames.
        """
        names=tuple(body_names)
        local=np.asarray(points_local,dtype=float)
        if not names or local.shape!=(len(names),3) or not np.isfinite(local).all():
            raise ValueError('one finite local point per nonempty body frame is required')
        frames=[self.model.getFrameId(name) for name in names]
        if any(frame>=len(self.model.frames) for frame in frames):
            raise ValueError('Unknown body frame')
        q=self.configuration(joint_position,**base)
        if with_jacobians:self.pin.computeJointJacobians(self.model,self.data,q)
        else:self.pin.forwardKinematics(self.model,self.data,q)
        self.pin.updateFramePlacements(self.model,self.data)
        points=np.empty((len(names),3))
        jacobians=np.empty((len(names),3,self.model.nv)) if with_jacobians else None
        for i,frame in enumerate(frames):
            pose=self.data.oMf[frame];offset=pose.rotation@local[i]
            points[i]=pose.translation+offset
            if with_jacobians:
                jac=np.asarray(self.pin.getFrameJacobian(self.model,self.data,frame,
                    self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)).reshape(6,self.model.nv)
                point_jac=jac[:3]+np.cross(jac[3:].T,offset).T
                jacobians[i]=point_jac[:,self.canonical_v_indices]
        return points,jacobians

    def point_after_integration(self, joint_position, tangent, body_name, point_local=(0, 0, 0), **base):
        q = self.pin.integrate(self.model, self.configuration(joint_position, **base), self._to_pin(tangent))
        _, pose = self._frame_pose(q, body_name)
        return pose.translation + pose.rotation @ _vec_array(point_local, 3)

    def manifest(self):
        return dict(source_path=self.source_path, source_sha256=self.source_sha256, pinocchio_version=self.pin.__version__, floating_base=self.floating_base, nq=self.model.nq, nv=self.model.nv, joint_names=list(self.joint_names), traversal_joint_names=self._traversal_joints, canonical_to_pin_v=self.canonical_v_indices.tolist(), total_mass_kg=self.total_mass, source_total_mass_kg=self.source_total_mass, body_count=self.body_count, source_damping=self.source_damping.tolist(), source_armature=self.source_armature.tolist(), effort_limits_Nm=self.effort_limits.tolist(), base_twist_frame="Pinocchio body-local linear,angular", euler_sequence=self.eulerseq, angle_unit=self.angle, joint_records=self._joint_records, body_records=self.body_records, keyframes=self.keyframes, contacts_included=False, friction_included=False, damping_included_in_bias_by_default=True, armature_included_in_mass_by_default=True)


def _vec_array(value, size):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected finite array of shape ({size},)")
    return result


def build_model(mjcf_path, floating_base=False, joint_names=JOINT_NAMES):
    return PhysicalModel(mjcf_path, floating_base=floating_base, joint_names=joint_names)
