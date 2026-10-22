"""Independent source-MJCF compiler cross-check, CPU Slurm only.

MuJoCo is isolated in model/mujoco_deps; shared IsaacSim environment is untouched.
This checks source parsing, not agreement with the converted USD/PhysX plant.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from v16_physical_model import JOINT_NAMES, NOMINAL_STAND, build_model


def compare(source, output):
    mj = mujoco.MjModel.from_xml_path(source)
    data = mujoco.MjData(mj)
    free = build_model(source, floating_base=True)
    fixed = build_model(source)
    mj_ids = np.array([mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES])
    q_indices = mj.jnt_qposadr[mj_ids]
    v_indices = mj.jnt_dofadr[mj_ids]
    perm = np.r_[np.arange(6), v_indices]
    rng = np.random.default_rng(1616)
    maxima = dict(mass_max_abs=0., bias_max_abs=0., fixed_mass_max_abs=0., fixed_gravity_max_abs=0., body_position_max_abs=0., body_rotation_max_abs=0., point_jacobian_max_abs=0.)
    raw_bias_coordinate_difference = 0.
    snapshots = []
    for sample in range(5):
        q = NOMINAL_STAND.copy() if sample == 0 else NOMINAL_STAND+rng.normal(0,.12,12)
        velocity = np.zeros(18) if sample == 0 else rng.normal(0,.15,18)
        data.qpos[:] = mj.qpos0
        data.qpos[q_indices] = q
        data.qvel[:] = 0
        data.qvel[perm] = velocity
        mujoco.mj_forward(mj, data)
        M_mj = np.zeros((18,18))
        mujoco.mj_fullM(mj, data, M_mj)
        M_mj = M_mj[np.ix_(perm,perm)]
        h_mj = data.qfrc_bias[perm].copy()
        M_pin = free.mass_matrix(q)
        h_pin = free.bias(q,velocity,include_damping=False)
        # Even at R=I, derivative of the moving local velocity frame is nonzero:
        # a_mj_world_linear = a_pin_local_linear + omega_local x v_local.
        # h_pin = h_mj + M * [omega x v_linear,0,...,0].
        transport_acceleration = np.zeros(18)
        transport_acceleration[:3] = np.cross(velocity[3:6],velocity[:3])
        h_pin_mj_coordinates = h_pin-M_pin@transport_acceleration
        raw_bias_coordinate_difference = max(raw_bias_coordinate_difference,float(np.max(np.abs(h_mj-h_pin))))
        maxima["mass_max_abs"] = max(maxima["mass_max_abs"],float(np.max(np.abs(M_mj-M_pin))))
        maxima["bias_max_abs"] = max(maxima["bias_max_abs"],float(np.max(np.abs(h_mj-h_pin_mj_coordinates))))
        maxima["fixed_mass_max_abs"] = max(maxima["fixed_mass_max_abs"],float(np.max(np.abs(M_mj[6:,6:]-fixed.mass_matrix(q)))))
        data.qvel[:] = 0
        mujoco.mj_forward(mj,data)
        maxima["fixed_gravity_max_abs"] = max(maxima["fixed_gravity_max_abs"],float(np.max(np.abs(data.qfrc_bias[v_indices]-fixed.gravity(q)))))
        for body in free.body_names:
            body_id = mujoco.mj_name2id(mj,mujoco.mjtObj.mjOBJ_BODY,body)
            _, pose = free._frame_pose(free.configuration(q),body)
            maxima["body_position_max_abs"] = max(maxima["body_position_max_abs"],float(np.max(np.abs(pose.translation-data.xpos[body_id]))))
            maxima["body_rotation_max_abs"] = max(maxima["body_rotation_max_abs"],float(np.max(np.abs(pose.rotation-data.xmat[body_id].reshape(3,3)))))
        for leg in ("FL","FR","RL","RR"):
            body_id = mujoco.mj_name2id(mj,mujoco.mjtObj.mjOBJ_BODY,leg+"_foot")
            point = data.xpos[body_id]+data.xmat[body_id].reshape(3,3)@np.array([.01,-.02,.03])
            jacp,jacr=np.zeros((3,18)),np.zeros((3,18))
            mujoco.mj_jac(mj,data,jacp,jacr,point,body_id)
            maxima["point_jacobian_max_abs"] = max(maxima["point_jacobian_max_abs"],float(np.max(np.abs(jacp[:,perm]-free.point_jacobian(q,leg+"_foot",[.01,-.02,.03])))))
        if sample == 0:
            snapshots.append(dict(joint_position=q.tolist(), mujoco_M=M_mj.tolist(), pinocchio_M=M_pin.tolist(), mujoco_bias=h_mj.tolist(), pinocchio_bias=h_pin.tolist()))
    mass_error=float(abs(np.sum(mj.body_mass)-free.total_mass))
    result=dict(slurm_job_id=os.environ.get("SLURM_JOB_ID"), source_sha256=free.source_sha256, mujoco_version=mujoco.__version__, pinocchio_version=free.pin.__version__, source_body_count=mj.nbody-1, source_nq=mj.nq, source_nv=mj.nv, sample_count=5, total_mass_kg=float(np.sum(mj.body_mass)), total_mass_error=mass_error, canonical_to_mujoco_v=perm.tolist(), maximum_errors=maxima, raw_bias_coordinate_difference_max_abs=raw_bias_coordinate_difference, nominal_snapshot=snapshots, conventions="Base orientation identity; velocity values coincide, but acceleration coordinates differ: a_mj_linear=a_pin_linear+omega cross v_linear. Bias comparison subtracts M times this transport acceleration from Pinocchio bias. Joint canonical mappings explicit.", scope="Source MJCF compiler comparison only; USD/PhysX comparison remains separate.")
    Path(output).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({key:value for key,value in result.items() if key != "nominal_snapshot"},indent=2))
    assert mass_error<1e-10, result
    assert all(error<2e-8 for error in maxima.values()),result
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--mjcf",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    compare(args.mjcf,args.output)
