"""Sensor-independent seated body references; +y forward, +x right."""
import numpy as np
from .motion import smooth


def dynamic_sit_reference(height_fraction, stand_height, height_drop,
                          pitch_rad=.4, rear_shift_m=.04):
    """Four-foot sit: transfer rearward, lower the rear and raise the front.

    Positive rotation about +x lifts the robot's +y front. The supervisor
    supplies a quintic height fraction; this function introduces an earlier
    rearward transfer and a gradual pitch. It never changes measured state.
    """
    if (not np.isfinite([height_fraction, stand_height, height_drop, pitch_rad, rear_shift_m]).all()
            or not 0 <= height_fraction <= 1 or not 0 <= height_drop <= .18
            or not 0 <= pitch_rad <= .5 or not 0 <= rear_shift_m <= .08):
        raise ValueError('Invalid dynamic seated reference')
    seated = 1.-float(height_fraction)
    return dict(height=float(stand_height-height_drop*seated),
                pitch_rad=float(pitch_rad*smooth(seated)),
                forward_offset_m=float(-rear_shift_m*smooth(1.4*seated)))


def posture_ready(gravity, velocity, angular_velocity, support_force,
                  height, target_height, pitch_rad=0.):
    values = np.r_[gravity, velocity, angular_velocity, support_force, height, target_height, pitch_rad]
    if not np.isfinite(values).all():
        return False
    target_gravity = np.array([0., -np.sin(pitch_rad), -np.cos(pitch_rad)])
    return bool(np.dot(gravity, target_gravity) > .97 and np.linalg.norm(velocity) < .12
                and np.linalg.norm(angular_velocity) < .35 and np.count_nonzero(np.asarray(support_force)>5.) == 4
                and abs(height-target_height) < .035)
