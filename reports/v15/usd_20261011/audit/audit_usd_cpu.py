#!/usr/bin/env python3
"""Read-only USD articulation audit. Requires pxr and numpy, never SimulationApp."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def plain(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (Gf.Quatf, Gf.Quatd)):
        return [float(value.GetReal()), *map(float, value.GetImaginary())]
    try:
        return [plain(item) for item in value]
    except TypeError:
        return str(value)


def matrix(value):
    return np.asarray(value, dtype=float).T


def frame(position, quaternion):
    result = np.eye(4)
    result[:3, :3] = matrix(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(quaternion))))
    result[:3, 3] = np.asarray(position, dtype=float)
    return result


def axis_rotation(axis, angle):
    result = np.eye(4)
    result[:3, :3] = matrix(Gf.Matrix3d(Gf.Rotation(Gf.Vec3d(*axis), math.degrees(angle))))
    return result


def bbox(bbox_cache, prim):
    value = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return {"min_m": plain(value.GetMin()), "max_m": plain(value.GetMax())}


def audit(path):
    stage = Usd.Stage.Open(path)
    if not stage:
        raise ValueError(f"Cannot compose {path}")
    transforms = UsdGeom.XformCache()
    boxes = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"], useExtentsHint=False)
    prims = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))
    bodies = {}
    joints = {}
    colliders = []
    for prim in prims:
        key = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass = UsdPhysics.MassAPI(prim)
            world = matrix(transforms.GetLocalToWorldTransform(prim))
            center = np.asarray(mass.GetCenterOfMassAttr().Get(), dtype=float)
            inertia = np.asarray(mass.GetDiagonalInertiaAttr().Get(), dtype=float)
            principal = mass.GetPrincipalAxesAttr().Get()
            if principal:
                rotation = frame([0, 0, 0], principal)[:3, :3]
                tensor = rotation @ np.diag(inertia) @ rotation.T
            else:
                tensor = np.diag(inertia)
            bodies[key] = {
                "name": prim.GetName(), "mass_kg": mass.GetMassAttr().Get(),
                "center_of_mass_local_m": plain(center),
                "center_of_mass_world_m": plain((world @ np.r_[center, 1])[:3]),
                "diagonal_inertia_kg_m2": plain(inertia), "principal_axes_wxyz": plain(principal),
                "inertia_tensor_local_kg_m2": plain(tensor), "world_transform": plain(world),
                "world_bounds": bbox(boxes, prim),
                "inertia_positive": bool((inertia > 0).all()),
                "inertia_triangle_inequality": bool(inertia.max() <= inertia.sum() - inertia.max() + 1e-9),
            }
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            body0 = joint.GetBody0Rel().GetTargets()
            body1 = joint.GetBody1Rel().GetTargets()
            if not body0 or not body1:
                continue
            body0, body1 = str(body0[0]), str(body1[0])
            world0 = matrix(transforms.GetLocalToWorldTransform(stage.GetPrimAtPath(body0)))
            world1 = matrix(transforms.GetLocalToWorldTransform(stage.GetPrimAtPath(body1)))
            local0 = frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
            local1 = frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
            zero0, zero1 = world0 @ local0, world1 @ local1
            error = np.linalg.inv(zero0) @ zero1
            angular_error = math.acos(np.clip((np.trace(error[:3, :3]) - 1) / 2, -1, 1))
            data = {
                "name": prim.GetName(), "type": prim.GetTypeName(), "body0": body0, "body1": body1,
                "local_frame0": plain(local0), "local_frame1": plain(local1),
                "zero_pivot_world_m": plain(zero0[:3, 3]),
                "authored_zero_position_error_m": float(np.linalg.norm(zero0[:3, 3] - zero1[:3, 3])),
                "authored_zero_rotation_error_rad": float(angular_error),
                "collision_enabled": joint.GetCollisionEnabledAttr().Get(),
            }
            if prim.IsA(UsdPhysics.RevoluteJoint):
                revolute = UsdPhysics.RevoluteJoint(prim)
                axis_name = revolute.GetAxisAttr().Get()
                axis = np.eye(3)[{"X": 0, "Y": 1, "Z": 2}[axis_name]]
                data.update(axis=axis_name, axis_local=plain(axis),
                            axis_world=plain(zero0[:3, :3] @ axis),
                            lower_limit_deg=revolute.GetLowerLimitAttr().Get(),
                            upper_limit_deg=revolute.GetUpperLimitAttr().Get(),
                            drive={name: plain(prim.GetAttribute("drive:angular:physics:" + name).Get())
                                   for name in ["type", "stiffness", "damping", "maxForce", "targetPosition", "targetVelocity"]})
            joints[key] = data
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            owner = prim.GetParent()
            while owner and not owner.IsPseudoRoot() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = owner.GetParent()
            colliders.append({
                "path": key, "owner": str(owner.GetPath()) if owner else None,
                "type": prim.GetTypeName(),
                "enabled": UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
                "approximation": prim.GetAttribute("physics:approximation").Get(),
                "world_bounds": bbox(boxes, prim),
            })
    masses = [(body["mass_kg"] or 0) for body in bodies.values()]
    total_mass = sum(masses)
    centroid = sum((body["mass_kg"] or 0) * np.asarray(body["center_of_mass_world_m"]) for body in bodies.values()) / total_mass
    body1_paths = {data["body1"] for data in joints.values()}
    root_paths = sorted(set(bodies) - body1_paths)
    feet = {}
    for body_path, body in bodies.items():
        if body["name"].endswith("_foot"):
            matching = [data for data in colliders if data["owner"] == body_path and data["enabled"]]
            if matching:
                lows = np.asarray([data["world_bounds"]["min_m"] for data in matching])
                highs = np.asarray([data["world_bounds"]["max_m"] for data in matching])
                low, high = lows.min(0), highs.max(0)
                feet[body["name"][:2]] = {"body": body_path, "collider": matching[0]["path"], "min_m": plain(low), "max_m": plain(high),
                                             "center_m": plain((low + high) / 2)}
    for data in colliders:
        if "/tennis_" in data["path"]:
            low = np.asarray(data["world_bounds"]["min_m"])
            high = np.asarray(data["world_bounds"]["max_m"])
            feet[data["owner"].rsplit("/", 1)[-1][:2]] = {
                "body": data["owner"], "collider": data["path"],
                "min_m": plain(low), "max_m": plain(high), "center_m": plain((low + high) / 2)}
    root_transform = np.asarray(bodies[root_paths[0]]["world_transform"])
    for foot in feet.values():
        foot["center_root_m"] = plain((np.linalg.inv(root_transform) @ np.r_[foot["center_m"], 1])[:3])
        foot["min_root_z_m"] = float(foot["min_m"][2] - root_transform[2, 3])
    return {
        "source": str(path), "usd_version": list(Usd.GetVersion()),
        "root_layer_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "layers": [{"identifier": layer.identifier, "real_path": layer.realPath} for layer in stage.GetUsedLayers()],
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage), "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "rigid_body_count": len(bodies), "revolute_joint_count": sum(data["type"] == "PhysicsRevoluteJoint" for data in joints.values()),
        "fixed_joint_count": sum(data["type"] == "PhysicsFixedJoint" for data in joints.values()),
        "total_mass_kg": total_mass, "center_of_mass_world_m": plain(centroid),
        "root_bodies": root_paths, "bodies": bodies, "joints": joints, "feet": feet,
        "colliders": colliders, "enabled_collider_count": sum(bool(data["enabled"]) for data in colliders),
        "articulation_roots": [str(prim.GetPath()) for prim in prims if prim.HasAPI(UsdPhysics.ArticulationRootAPI)],
        "cameras": [str(prim.GetPath()) for prim in prims if prim.IsA(UsdGeom.Camera)],
        "sensor_named_prims": [str(prim.GetPath()) for prim in prims if any(word in prim.GetName().lower() for word in ["imu", "sensor", "camera"])],
    }


def fk(data, targets):
    stage = Usd.Stage.Open(data["source"])
    transforms = UsdGeom.XformCache()
    poses = {path: np.asarray(data["bodies"][path]["world_transform"]) for path in data["root_bodies"]}
    pending = list(data["joints"].values())
    while pending:
        progress = False
        for joint in pending[:]:
            if joint["body0"] not in poses:
                continue
            angle = targets.get(joint["name"], 0)
            rotation = axis_rotation(joint.get("axis_local", [1, 0, 0]), angle)
            poses[joint["body1"]] = poses[joint["body0"]] @ np.asarray(joint["local_frame0"]) @ rotation @ np.linalg.inv(np.asarray(joint["local_frame1"]))
            pending.remove(joint)
            progress = True
        if not progress:
            raise ValueError("Joint graph cannot resolve: " + str([joint["name"] for joint in pending]))
    feet = {}
    root_inverse = np.linalg.inv(poses[data["root_bodies"][0]])
    for leg, foot in data["feet"].items():
        old_pose = np.asarray(data["bodies"][foot["body"]]["world_transform"])
        local_point = np.linalg.inv(old_pose) @ np.r_[foot["center_m"], 1]
        world_point = poses[foot["body"]] @ local_point
        mesh_prim = stage.GetPrimAtPath(foot["collider"])
        vertices = np.asarray(UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get(), dtype=float)
        geometry_world = matrix(transforms.GetLocalToWorldTransform(mesh_prim))
        geometry_body = np.linalg.inv(old_pose) @ geometry_world
        current_geometry_world = poses[foot["body"]] @ geometry_body
        points_world = (current_geometry_world @ np.c_[vertices, np.ones(len(vertices))].T).T[:, :3]
        points_root = (root_inverse @ np.c_[points_world, np.ones(len(vertices))].T).T[:, :3]
        feet[leg] = {"center_world_m": plain(world_point[:3]), "center_root_m": plain((root_inverse @ world_point)[:3]),
                     "mesh_vertex_min_root_m": plain(points_root.min(axis=0)),
                     "mesh_vertex_max_root_m": plain(points_root.max(axis=0)), "mesh_vertex_count": len(vertices)}
    return {"feet": feet, "resolved_bodies": len(poses)}


def compare(old, new):
    old_joints = {data["name"]: data for data in old["joints"].values() if "axis_world" in data}
    new_joints = {data["name"]: data for data in new["joints"].values() if "axis_world" in data}
    mapping = []
    defaults_old, defaults_new = {}, {}
    for old_leg, new_leg in [("FL", "FL"), ("FR", "FR"), ("RL", "BL"), ("RR", "BR")]:
        for canonical, native, default in [("HAA", "hip_roll", 0), ("HFE", "hip_pitch", -0.55), ("KFE", "knee", 1.10)]:
            old_name, new_name = old_leg + "_" + canonical, new_leg + "_" + native
            old_axis = np.asarray(old_joints[old_name]["axis_world"])
            new_axis = np.asarray(new_joints[new_name]["axis_world"])
            similarity = float(np.dot(old_axis, new_axis) / np.linalg.norm(old_axis) / np.linalg.norm(new_axis))
            sign = 1 if similarity >= 0 else -1
            old_root = np.asarray(old["bodies"][old["root_bodies"][0]]["world_transform"])[:3, 3]
            new_root = np.asarray(new["bodies"][new["root_bodies"][0]]["world_transform"])[:3, 3]
            pivot_delta = (np.asarray(new_joints[new_name]["zero_pivot_world_m"]) - new_root
                           - np.asarray(old_joints[old_name]["zero_pivot_world_m"]) + old_root)
            axis_normalized = old_axis / np.linalg.norm(old_axis)
            mapping.append({"canonical": old_name, "native": new_name, "native_sign": sign,
                            "axis_cosine": similarity, "native_default_rad": default * sign,
                            "pivot_delta_root_m": plain(pivot_delta),
                            "joint_axis_line_offset_m": float(np.linalg.norm(pivot_delta - np.dot(pivot_delta, axis_normalized) * axis_normalized)),
                            "pivot_delta_world_m": plain(np.asarray(new_joints[new_name]["zero_pivot_world_m"]) - np.asarray(old_joints[old_name]["zero_pivot_world_m"]))})
            defaults_old[old_name], defaults_new[new_name] = default, default * sign
    return {"convention": "q_native = native_sign * q_canonical; identical mapping for dq and torque",
            "mapping": mapping, "canonical_stand_rad": defaults_old, "native_stand_rad": defaults_new,
            "old_zero_fk": fk(old, {}), "new_zero_fk": fk(new, {}),
            "old_stand_fk": fk(old, defaults_old), "new_stand_fk": fk(new, defaults_new),
            "mass_ratio_new_over_old": new["total_mass_kg"] / old["total_mass_kg"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    old, new = audit(args.old), audit(args.new)
    result = {"scope": "CPU USD composition and analytic forward kinematics; no PhysX stepping or walking validation", "old": old, "new": new, "comparison": compare(old, new)}
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps({key: {name: result[key][name] for name in ["rigid_body_count", "revolute_joint_count", "fixed_joint_count", "total_mass_kg", "enabled_collider_count", "articulation_roots", "cameras"]} for key in ["old", "new"]}, indent=2))
    print(json.dumps(result["comparison"], indent=2))


if __name__ == "__main__":
    main()
