"""Compare one PhysX snapshot against source MJCF, on HPC CPU only.

No simulator is launched. All public joint arrays are canonical. Tensor SDK
107.3 Articulation.get_inertias uses COM-referenced inertia in actor/prim axes,
packed column-major. Root generalized velocity is world COM linear/angular.
The CLI requires an actual probe snapshot; importing helpers performs no model
calculation. Pure coordinate algebra tests can be run locally.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

JOINT_NAMES=tuple(f"{leg}_{joint}" for leg in ("FL","FR","RL","RR") for joint in ("HAA","HFE","KFE"))
FOOT_NAMES=tuple(leg+"_foot" for leg in ("FL","FR","RL","RR"))
SDK_REFERENCE="https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html"


def skew(vector):
    x,y,z=np.asarray(vector,float)
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def quaternion_matrix(wxyz):
    w,x,y,z=np.asarray(wxyz,float)/np.linalg.norm(wxyz)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def base_velocity_transform(actor_rotation,offset_world,joint_count=12):
    """v_COM_world = T v_actor_local, including omega cross actor-to-COM."""
    R=np.asarray(actor_rotation,float)
    T=np.eye(6+joint_count)
    T[:3,:3]=R
    T[:3,3:6]=-skew(offset_world)@R
    T[3:6,3:6]=R
    return T


def transport_acceleration(actor_rotation,actor_velocity_local,root_com_offset_actor,joint_count=12):
    """Tdot*v for actor-local twist to world COM velocity derivatives."""
    velocity=np.asarray(actor_velocity_local,float)
    omega=velocity[3:6]
    result=np.zeros(6+joint_count)
    result[:3]=np.asarray(actor_rotation)@(np.cross(omega,velocity[:3])+np.cross(omega,np.cross(omega,root_com_offset_actor)))
    return result


def normalize_dynamics(M_world,h_world,T,transport):
    """Mass and bias in Pin actor-local coordinates; forces transform dually."""
    M=np.asarray(M_world,float)
    return T.T@M@T,T.T@(np.asarray(h_world,float)+M@transport)


def unpack_inertias(packed):
    values=np.asarray(packed,float)
    if values.ndim==2 and values.shape[1]==9:
        return np.asarray([row.reshape(3,3,order="F") for row in values])
    if values.ndim==3 and values.shape[1:]==(3,3):
        return values.copy()
    raise ValueError("Expected PhysX inertia packing [body,9] column-major")


def point_in_source_frame(source_position,source_rotation,world_point):
    """Represent the measured physical point in the source actor frame."""
    return np.asarray(source_rotation,float).T@(np.asarray(world_point,float)-np.asarray(source_position,float))


def aggregate_mass_properties(masses,coms_world,inertias_com_world):
    masses,positions,inertias=(np.asarray(value,float) for value in (masses,coms_world,inertias_com_world))
    total=float(np.sum(masses))
    if total<=0:
        raise ValueError("Nonpositive aggregate mass")
    com=np.sum(masses[:,None]*positions,axis=0)/total
    inertia=np.zeros((3,3))
    for mass,position,tensor in zip(masses,positions,inertias):
        offset=position-com
        inertia+=tensor+mass*((offset@offset)*np.eye(3)-np.outer(offset,offset))
    return total,com,inertia


def _metric(actual,reference):
    actual,reference=np.asarray(actual,float),np.asarray(reference,float)
    if actual.shape!=reference.shape:
        raise ValueError(f"Comparison shape mismatch: {actual.shape} vs {reference.shape}")
    difference=actual-reference
    return dict(shape=list(actual.shape),max_abs=float(np.max(np.abs(difference))),frobenius=float(np.linalg.norm(difference)),relative_frobenius=float(np.linalg.norm(difference)/max(np.linalg.norm(reference),1e-14)))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _field(snapshot,*names,required=True):
    for name in names:
        if name in snapshot:
            value=np.asarray(snapshot[name])
            if value.dtype.kind not in "USO" and not np.isfinite(value).all():
                raise ValueError(f"Nonfinite snapshot field: {name}")
            return value
    if required:
        raise ValueError(f"Required snapshot field is missing: {names}")
    return None


def _metadata(manifest,*names,default=None):
    for name in names:
        if name in manifest:
            return manifest[name]
    conventions=manifest.get("tensor_conventions",{})
    for name in names:
        if name in conventions:
            return conventions[name]
    return default


def _full_frame_jacobian(model,q,body,point_local,base):
    native=model.configuration(q,**base)
    frame,pose=model._frame_pose(native,body)
    rotation=pose.rotation.copy()
    J=np.asarray(model.pin.computeFrameJacobian(model.model,model.data,native,frame,model.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)).reshape(6,18)
    J[:3]-=skew(rotation@point_local)@J[3:]
    return J[:,model.canonical_v_indices]


def compare(probe_dir,mjcf,output_prefix,*,inertia_frame="actor",armature_in_mass="unknown"):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Actual physical-model comparison requires an HPC Slurm CPU job")
    probe_dir=Path(probe_dir)
    nominal=probe_dir/"physx_nominal.npz"
    manifest_path=probe_dir/"manifest.json"
    manifest=json.loads(manifest_path.read_text())
    data=np.load(nominal,allow_pickle=False)
    names=tuple(str(name) for name in _field(data,"joint_names"))
    if set(names)!=set(JOINT_NAMES) or len(names)!=12:
        raise ValueError("Probe joints do not match source canonical joints")
    joint_perm=np.array([names.index(name) for name in JOINT_NAMES])
    generalized_perm=np.r_[np.arange(6),6+joint_perm]
    q=_field(data,"q")[joint_perm].astype(float)
    dq=_field(data,"dq","joint_velocities")[joint_perm].astype(float)
    actor_position=_field(data,"root_actor_position_w","root_actor_position").astype(float)
    actor_quaternion=_field(data,"root_actor_quaternion_wxyz","root_quaternion_wxyz").astype(float)
    com_position=_field(data,"root_com_position_w","root_com_position").astype(float)
    actor_rotation=quaternion_matrix(actor_quaternion)
    actor_twist=_field(data,"root_actor_twist_w",required=False)
    if actor_twist is None:
        actor_linear=_field(data,"root_actor_linear_velocity_w","root_actor_linear_velocity",required=False)
        angular=_field(data,"root_angular_velocity_w","root_angular_velocity")
        if actor_linear is not None:
            actor_twist=np.r_[actor_linear,angular]
    com_twist=_field(data,"root_com_twist_w",required=False)
    if com_twist is None:
        com_twist=np.r_[_field(data,"root_com_linear_velocity_w","root_com_linear_velocity"),_field(data,"root_angular_velocity_w","root_angular_velocity")]
    offset_world=com_position-actor_position
    offset_actor=actor_rotation.T@offset_world
    if actor_twist is None:
        actor_twist=np.r_[com_twist[:3]-np.cross(com_twist[3:],offset_world),com_twist[3:]]
        actor_twist_origin="derived from measured COM world twist and actor-to-COM offset"
    else:
        actor_twist_origin="exported actor world twist"
    base_local=np.r_[actor_rotation.T@actor_twist[:3],actor_rotation.T@actor_twist[3:]]
    velocity=np.r_[base_local,dq]
    T=base_velocity_transform(actor_rotation,offset_world)
    transport=transport_acceleration(actor_rotation,base_local,offset_actor)
    twist_consistency=_metric((T@velocity)[:6],com_twist)
    M_px=_field(data,"M")[np.ix_(generalized_perm,generalized_perm)].astype(float)
    gravity_px=_field(data,"gravity","gravity_forces","g")[generalized_perm].astype(float)
    coriolis_px=_field(data,"coriolis","coriolis_forces")[generalized_perm].astype(float)
    h_px=_field(data,"h")[generalized_perm].astype(float)
    M_normalized,h_normalized=normalize_dynamics(M_px,h_px,T,transport)
    gravity_normalized=T.T@gravity_px
    coriolis_normalized=T.T@(coriolis_px+M_px@transport)
    body_names=tuple(str(name) for name in _field(data,"body_names"))
    positions=_field(data,"body_positions","body_actor_positions_w").astype(float)
    quaternions=_field(data,"body_quaternions","body_actor_quaternions_wxyz").astype(float)
    rotations=np.array([quaternion_matrix(quat) for quat in quaternions])
    masses=_field(data,"masses").astype(float)
    coms=_field(data,"coms").astype(float)
    inertias=unpack_inertias(_field(data,"inertias"))
    if positions.shape!=(len(body_names),3) or quaternions.shape!=(len(body_names),4) or coms.shape!=(len(body_names),7):
        raise ValueError("Body snapshot poses/COM packing mismatch")
    body_coms_world=positions+np.einsum("bij,bj->bi",rotations,coms[:,:3])
    com_local_rotations=np.array([quaternion_matrix(quat[[3,0,1,2]]) for quat in coms[:,3:]])
    inertia_rotations=rotations if inertia_frame=="actor" else rotations@com_local_rotations
    inertia_world=inertia_rotations@inertias@np.transpose(inertia_rotations,(0,2,1))
    exported_body_com=_field(data,"body_com_positions_w","body_com_positions",required=False)
    source_config=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(source_config))
    from v16_physical_model import build_model
    model=build_model(mjcf,floating_base=True)
    source_gravity=model.model.gravity.linear.copy()
    runtime_gravity=_field(data,"gravity_w","gravity_vector_w","gravity_world",required=False)
    if runtime_gravity is not None:
        model.model.gravity.linear=np.asarray(runtime_gravity,float)
    base=dict(base_position=actor_position,base_quaternion_xyzw=actor_quaternion[[1,2,3,0]])
    M_rigid=model.mass_matrix(q,include_armature=False,**base)
    M_armature=model.mass_matrix(q,include_armature=True,**base)
    g_source=model.gravity(q,**base)
    h_source=model.bias(q,velocity,include_damping=False,**base)
    c_source=h_source-g_source
    arrays=dict(q=q,dq=dq,velocity_actor_local=velocity,T_actor_local_to_com_world=T,Tdot_velocity=transport,
                M_physx_world_com=M_px,M_physx_actor_local=M_normalized,M_source_rigid=M_rigid,
                M_source_with_armature=M_armature,g_source=g_source,g_physx_actor_local=gravity_normalized,
                coriolis_source=c_source,coriolis_physx_actor_local=coriolis_normalized,
                h_source_rigid=h_source,h_source_with_damping=h_source+model.damping_generalized*velocity,
                h_physx_actor_local=h_normalized,body_names_physx=np.array(body_names),
                positions_physx=positions,coms_world_physx=body_coms_world,inertias_world_physx=inertia_world)
    dynamics=dict(M_raw_vs_source_rigid=_metric(M_normalized,M_rigid),M_raw_vs_source_with_armature=_metric(M_normalized,M_armature),
                  gravity=_metric(gravity_normalized,g_source),coriolis_with_transport=_metric(coriolis_normalized,c_source),
                  bias_with_transport=_metric(h_normalized,h_source),bias_without_transport=_metric(T.T@h_px,h_source),
                  h_vs_exported_gravity_plus_coriolis=_metric(h_px,gravity_px+coriolis_px))
    for mode,reference in (("rigid",M_rigid),("source_armature",M_armature)):
        dynamics["M_blocks_"+mode]={label:_metric(M_normalized[np.ix_(indices,indices)],reference[np.ix_(indices,indices)]) for label,indices in (("base_translation",np.arange(3)),("base_rotation",np.arange(3,6)),("joint_block",np.arange(6,18)))}
        dynamics["M_base_joint_coupling_"+mode]=_metric(M_normalized[:6,6:],reference[:6,6:])
    armature=_field(data,"joint_armatures","joint_armature","armatures",required=False)
    damping=_field(data,"joint_damping","damping",required=False)
    friction=_field(data,"joint_friction","friction",required=False)
    if armature is not None:
        armature=np.asarray(armature)[joint_perm]
        arrays["armature_physx"]=armature
        if armature_in_mass=="included":
            M_px_rigid=M_normalized-np.diag(np.r_[np.zeros(6),armature])
            arrays["M_physx_armature_removed"]=M_px_rigid
            dynamics["M_measured_armature_removed_vs_source_rigid"]=_metric(M_px_rigid,M_rigid)
    source_records={record["name"]:record for record in model.body_records}
    source_positions,source_coms,source_inertias,source_rotations=[],[],[],[]
    for body in model.body_names:
        _,pose=model._frame_pose(model.configuration(q,**base),body)
        record=source_records[body]
        source_positions.append(pose.translation.copy())
        source_rotations.append(pose.rotation.copy())
        source_coms.append(pose.translation+pose.rotation@np.array(record["com_body"]))
        source_inertias.append(pose.rotation@np.array(record["inertia_com_body"])@pose.rotation.T)
    arrays.update(body_names_source=np.array(model.body_names),positions_source=np.array(source_positions),
                  coms_world_source=np.array(source_coms),inertias_world_source=np.array(source_inertias))
    body_actor_twists=_field(data,"body_actor_twists_w",required=False)
    body_com_twists=_field(data,"body_com_twists_w",required=False)
    body_com_quaternions=_field(data,"body_com_quaternions_wxyz",required=False)
    body_com_rotations=rotations@com_local_rotations
    actor_twist_errors=[]
    com_twist_errors=[]
    body_rows=[]
    groups={}
    unmatched=[]
    for index,body in enumerate(body_names):
        if body not in source_records:
            unmatched.append(body)
            continue
        record=source_records[body]
        source_index=model.body_names.index(body)
        group=int(record["parent_pin_joint"])
        groups.setdefault(group,[]).append(index)
        body_rows.append(dict(body=body,source_pin_joint=group,physx_mass_kg=float(masses[index]),
                              source_individual_mass_kg=record["mass"],actor_position_error_m=float(np.max(np.abs(positions[index]-source_positions[source_index]))),
                              actor_rotation_matrix_error=float(np.max(np.abs(rotations[index]-source_rotations[source_index]))),
                              individual_com_world_error_m=float(np.max(np.abs(body_coms_world[index]-source_coms[source_index]))),
                              individual_inertia_world_max_abs=float(np.max(np.abs(inertia_world[index]-source_inertias[source_index])))))
        if body_actor_twists is not None:
            point=point_in_source_frame(source_positions[source_index],source_rotations[source_index],positions[index])
            prediction=_full_frame_jacobian(model,q,body,point,base)@velocity
            actor_twist_errors.append(float(np.max(np.abs(prediction-body_actor_twists[index]))))
        if body_com_twists is not None:
            point=point_in_source_frame(source_positions[source_index],source_rotations[source_index],body_coms_world[index])
            prediction=_full_frame_jacobian(model,q,body,point,base)@velocity
            com_twist_errors.append(float(np.max(np.abs(prediction-body_com_twists[index]))))
    model.pin.forwardKinematics(model.model,model.data,model.configuration(q,**base))
    clusters=[]
    for group,indices in sorted(groups.items()):
        px_mass,px_com,px_inertia=aggregate_mass_properties(masses[indices],body_coms_world[indices],inertia_world[indices])
        source_inertia=model.model.inertias[group]
        pose=model.data.oMi[group]
        source_com=pose.translation+pose.rotation@source_inertia.lever
        source_tensor=pose.rotation@source_inertia.inertia@pose.rotation.T
        clusters.append(dict(source_pin_joint=group,source_joint_name=str(model.model.names[group]),
                             physx_members=[body_names[index] for index in indices],source_fixed_members=[body for body in model.body_names if source_records[body]["parent_pin_joint"]==group],
                             physx_mass_kg=px_mass,source_mass_kg=float(source_inertia.mass),mass_difference_kg=px_mass-float(source_inertia.mass),
                             com_world_error_m=float(np.max(np.abs(px_com-source_com))),inertia_com_world=_metric(px_inertia,source_tensor)))
    J_px=_field(data,"J_feet")[...,generalized_perm].astype(float)
    validation_path=probe_dir/"jacobian_validation.json"
    validation=json.loads(validation_path.read_text()) if validation_path.exists() else {}
    jacobian_origin=validation.get("selected_origin",_metadata(manifest,"jacobian_link_origin",default="unvalidated"))
    if jacobian_origin not in ("actor","com"):
        raise ValueError("Runtime Jacobian link origin must be characterized as actor or com")
    J_source=[]
    J_source_same_local=[]
    base_hypotheses={}
    for foot_index,foot in enumerate(FOOT_NAMES):
        index=body_names.index(foot)
        point_local=coms[index,:3] if jacobian_origin=="com" else np.zeros(3)
        source_index=model.body_names.index(foot)
        target_world=body_coms_world[index] if jacobian_origin=="com" else positions[index]
        same_world_local=point_in_source_frame(source_positions[source_index],source_rotations[source_index],target_world)
        J_source.append(_full_frame_jacobian(model,q,foot,same_world_local,base))
        J_source_same_local.append(_full_frame_jacobian(model,q,foot,point_local,base))
    J_source=np.array(J_source)
    J_source_same_local=np.array(J_source_same_local)
    J_normalized=J_px@T
    for root_label,root_point in (("root_com",com_position),("root_actor",actor_position)):
        for link_label in ("com","actor"):
            expected=[]
            for foot in FOOT_NAMES:
                index=body_names.index(foot)
                point=body_coms_world[index] if link_label=="com" else positions[index]
                expected.append(np.block([[np.eye(3),-skew(point-root_point)],[np.zeros((3,3)),np.eye(3)]]))
            base_hypotheses[root_label+"_world_to_link_"+link_label]=_metric(J_px[...,:6],np.array(expected))
    arrays.update(J_feet_physx_world_com=J_px,J_feet_physx_actor_local=J_normalized,J_feet_source_actor_local=J_source,J_feet_source_same_local_coordinates=J_source_same_local)
    report=dict(schema_version=1,slurm_job_id=os.environ["SLURM_JOB_ID"],probe_dir=str(probe_dir.resolve()),
                probe_npz_sha256=_sha(nominal),probe_manifest_sha256=_sha(manifest_path),
                comparison_code_sha256=_sha(__file__),source_sha256=model.source_sha256,
                source_code_sha256=_sha(source_config/"v16_physical_model.py"),usd_path=manifest.get("usd_path"),usd_sha256=manifest.get("usd_sha256"),
                sdk_contract_reference=SDK_REFERENCE,joint_order=list(JOINT_NAMES),canonical_to_probe_joint_indices=joint_perm.tolist(),
                source_total_mass_kg=model.total_mass,physx_total_mass_kg=float(np.sum(masses)),total_mass_difference_kg=float(np.sum(masses)-model.total_mass),
                source_body_count=model.body_count,physx_body_count=len(body_names),unmatched_physx_bodies=unmatched,
                source_gravity_world=source_gravity.tolist(),comparison_gravity_world=model.model.gravity.linear.tolist(),
                coordinate_conditions=dict(root_input="PhysX root COM world linear/angular",source_input="Pin actor/body-local linear/angular",actor_twist_origin=actor_twist_origin,
                                           root_actor_gauge="Source base actor pose is assigned the measured PhysX root actor pose; common base actor coordinates are the declared correspondence, not fitted from dynamics",
                                           bias_transport="Tdot v linear = R*(omega cross v_actor + omega cross (omega cross r_actor_COM)); h_Pin=T^T*(h_PX+M_PX*Tdot v)",
                                           inertia_reference="link COM",inertia_axes=inertia_frame,inertia_packing="column-major9",jacobian_link_origin=jacobian_origin,
                                           jacobian_rows="world linear, angular",jacobian_point="Same measured PhysX world COM/actor point represented in the source actor frame; same-local-coordinate comparison retained only as raw diagnostic",
                                           source_armature_removed_for_rigid_mode=True,source_damping_removed_from_bias=True,source_contact_and_friction_included=False,
                                           physx_armature_in_mass=armature_in_mass,physx_bias_excludes=manifest.get("model_compensation_excludes",["contact","joint friction","damping"])),
                measured_physx_joint_armature=None if armature is None else armature.tolist(),measured_physx_joint_damping=None if damping is None else np.asarray(damping)[joint_perm].tolist(),
                measured_physx_joint_friction=None if friction is None else np.asarray(friction)[joint_perm].tolist(),
                source_joint_armature=model.source_armature.tolist(),source_joint_damping=model.source_damping.tolist(),
                root_twist_consistency=twist_consistency,body_com_pose_consistency=None if exported_body_com is None else _metric(body_coms_world,exported_body_com),
                body_com_rotation_consistency=None if body_com_quaternions is None else _metric(body_com_rotations,np.array([quaternion_matrix(quat) for quat in body_com_quaternions])),
                source_vs_measured_actor_twist_max_abs=None if not actor_twist_errors else max(actor_twist_errors),
                source_vs_measured_com_twist_max_abs=None if not com_twist_errors else max(com_twist_errors),
                dynamics=dynamics,jacobian=_metric(J_normalized,J_source),jacobian_same_local_coordinates_raw=_metric(J_normalized,J_source_same_local),jacobian_base_coordinate_hypotheses=base_hypotheses,
                clusters=clusters,body_rows=body_rows,
                interpretation="Measured differences are retained. This report does not presume the converted USD matches the source MJCF.",
                limitations=["One settled snapshot; neither global dynamics equality nor walking stability is established.","If the USD root actor frame was reauthored relative to the source base actor, an additional independently measured root gauge transform is required before attributing all residuals to physical parameters.","Individual body mass/COM/inertia may differ when fixed members are merged; movable-joint cluster aggregates are the appropriate additional comparison.","Root COM-world coordinate basis is tested with the measured base Jacobian block; actor/world alternative hypotheses are reported rather than selected silently.","Armature is removed from PhysX mass only when inclusion is explicitly declared and measured armature values exist."])
    prefix=Path(output_prefix)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    np.savez_compressed(prefix.with_suffix(".npz"),**arrays)
    with prefix.with_name(prefix.name+"_bodies.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(body_rows[0]) if body_rows else ["body"])
        writer.writeheader()
        writer.writerows(body_rows)
    print(json.dumps({key:report[key] for key in ("slurm_job_id","source_total_mass_kg","physx_total_mass_kg","total_mass_difference_kg","source_body_count","physx_body_count","root_twist_consistency","dynamics","jacobian","jacobian_base_coordinate_hypotheses")},indent=2))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--probe-dir",required=True)
    parser.add_argument("--mjcf",required=True)
    parser.add_argument("--output-prefix",required=True)
    parser.add_argument("--inertia-frame",choices=("actor","com"),default="actor")
    parser.add_argument("--physx-armature-in-mass",choices=("included","excluded","unknown"),default="unknown")
    args=parser.parse_args()
    compare(args.probe_dir,args.mjcf,args.output_prefix,inertia_frame=args.inertia_frame,armature_in_mass=args.physx_armature_in_mass)
