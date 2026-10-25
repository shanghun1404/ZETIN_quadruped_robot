"""V151 standing controller supplied with the real 13-body CAD dynamics.

The inherited interface consumes canonical joints, actor-origin base position,
actor-origin linear velocity in body axes, and wxyz quaternions. Virtual foot
frames add neither mass nor collision geometry to the source USD.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .standing import StandingController
from v16_controller_math import digital_acceleration_gains
from v16_physical_model import PhysicalModel


class CadStandingController(StandingController):
    def __init__(self, model_path, dt=.02, metadata_path=None):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive and finite')
        model_path = Path(model_path)
        metadata_path = Path(metadata_path) if metadata_path else model_path.with_name('cad_model.json')
        self.metadata = json.loads(metadata_path.read_text())
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != self.metadata['dynamics_mjcf_sha256']:
            raise ValueError('CAD dynamics model does not match its metadata')
        self.dt = float(dt)
        self.model = PhysicalModel(model_path, floating_base=True)
        if self.model.body_count != 13 or abs(self.model.total_mass - self.metadata['source_total_mass_kg']) > 1e-10:
            raise ValueError('CAD dynamics must retain the original 13 physical bodies and mass')
        self.nominal_joint_position = np.asarray(self.metadata['canonical_nominal_q'], float)
        self.nominal_height = float(self.metadata['nominal_root_height_m'])
        self.limits = np.asarray(self.metadata['effort_limits_Nm'], float)
        self.friction = float(self.metadata['wbc_friction_coefficient'])
        self.kp, self.kd = digital_acceleration_gains(20.0344, .8295, self.dt)
        self.foot_names = tuple(f'{leg}_foot' for leg in ('FL', 'FR', 'RL', 'RR'))
        pin = self.model.pin
        for name in self.foot_names:
            foot = self.metadata['foot_frames'][name]
            parent_id = self.model.model.getFrameId(foot['parent_body'])
            parent = self.model.model.frames[parent_id]
            placement = parent.placement * pin.SE3(np.eye(3), np.asarray(foot['center_body_m']))
            self.model.model.addFrame(pin.Frame(name, parent.parent, parent_id, placement, pin.FrameType.OP_FRAME))
        self.model.data = self.model.model.createData()
        self.contact_local = []
        base = self._base([0, 0, self.nominal_height], [1, 0, 0, 0])
        q = self.model.configuration(self.nominal_joint_position, **base)
        for name in self.foot_names:
            _, pose = self.model._frame_pose(q, name)
            self.contact_local.append(pose.rotation.T @ [0, 0, -pose.translation[2]])
        self.contact_local = np.asarray(self.contact_local)
        self.contact_anchors = None
        self._warm = None
        self._previous = np.zeros(12)
        self._previous_reference = None

    def contact_consistent_reference(self, joint_pos, base_pos, quat_wxyz, height,
                                     stance_mask, center_body=False, forward_offset_m=0.,
                                     landing_depth_m=0.):
        """Land at the CAD contact plane without demanding 5 mm penetration.

        Recorded handover state 2320/17.920s is unreachable with the inherited
        below-plane target, but reaches the plane with the same exact IK gate.
        Measured contacts still determine support and the subsequent handover.
        """
        return super().contact_consistent_reference(joint_pos, base_pos, quat_wxyz,
            height, stance_mask, center_body=center_body,
            forward_offset_m=forward_offset_m, landing_depth_m=landing_depth_m)


CADStandingController = CadStandingController
