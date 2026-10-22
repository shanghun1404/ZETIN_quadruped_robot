"""Read-only kinematic replay of the recorded stop handover; no simulator."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
MODEL = ROOT / "config/experiments/v15_cad_20261011/model"
sys.path.insert(0, str(ROOT / "config"))
from v16_physical_model import _axis_rotation, mjcf_rotation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RecordedGeometry:
    """FK from the explicit CAD MJCF, checked against its original USD audit."""
    def __init__(self):
        self.metadata = json.loads((MODEL / "cad_model.json").read_text())
        assert digest(MODEL / "quadruped_cad.xml") == self.metadata["dynamics_mjcf_sha256"]
        self.root = ET.parse(MODEL / "quadruped_cad.xml").getroot().find("worldbody/body")
        self.names = self.metadata["canonical_joint_names"]
        self.feet = list(self.metadata["foot_frames"].values())
        self.hulls = np.load(MODEL / "collision_hulls.npz", allow_pickle=False)
        assert str(self.hulls["usd_sha256"]) == self.metadata["source_usd_root_sha256"]
        self.hull_records = json.loads(str(self.hulls["records_json"]))
        self.foot_hull = [next(i for i, r in enumerate(self.hull_records)
                             if r["path"] == f["collider"]) for f in self.feet]
        nominal = self.poses(np.asarray(self.metadata["canonical_nominal_q"]),
                             [0, 0, self.metadata["nominal_root_height_m"]], [1, 0, 0, 0])
        self.material = []
        errors = []
        for foot in self.feet:
            rotation, position = nominal[foot["parent_body"]]
            center = rotation @ foot["center_body_m"] + position
            audit = np.asarray(foot["audited_nominal_center_root_m"]) + [0, 0, self.metadata["nominal_root_height_m"]]
            errors.append(np.max(np.abs(center - audit)))
            self.material.append(np.asarray(foot["center_body_m"]) + rotation.T @ [0, 0, -center[2]])
        self.nominal_audit_max_error_m = float(max(errors))
        assert self.nominal_audit_max_error_m < 1e-8

    def poses(self, joints, position, quaternion):
        values = dict(zip(self.names, joints))
        result = {}
        base_rotation = mjcf_rotation(ET.Element("body", quat=" ".join(map(str, quaternion))))

        def visit(body, rotation, origin, is_root=False):
            if not is_root:
                origin = origin + rotation @ np.fromstring(body.get("pos", "0 0 0"), sep=" ")
                rotation = rotation @ mjcf_rotation(body, angle="radian")
                joint = body.find("joint")
                if joint is not None:
                    anchor = np.fromstring(joint.get("pos", "0 0 0"), sep=" ")
                    delta = _axis_rotation(np.fromstring(joint.get("axis"), sep=" "), values[joint.get("name")])
                    origin = origin + rotation @ (anchor - delta @ anchor)
                    rotation = rotation @ delta
            result[body.get("name")] = (rotation, origin)
            for child in body.findall("body"):
                visit(child, rotation, origin)

        visit(self.root, base_rotation, np.asarray(position, dtype=float), True)
        return result

    def measure(self, joints, position, quaternion):
        poses = self.poses(joints, position, quaternion)
        points, minimum = [], []
        for foot, material, hull_index in zip(self.feet, self.material, self.foot_hull):
            rotation, origin = poses[foot["parent_body"]]
            # material includes the virtual foot-frame center and the nominal
            # projected contact offset, expressed in the parent-body frame.
            points.append(rotation @ material + origin)
            minimum.append(float((self.hulls[f"vertices_{hull_index}"] @ rotation.T + origin)[:, 2].min()))
        mass, weighted = 0., np.zeros(3)
        for body in self.root.iter("body"):
            inertial = body.find("inertial")
            value = float(inertial.get("mass"))
            rotation, origin = poses[body.get("name")]
            weighted += value * (rotation @ np.fromstring(inertial.get("pos"), sep=" ") + origin)
            mass += value
        assert abs(mass - self.metadata["source_total_mass_kg"]) < 1e-10
        return np.asarray(points), np.asarray(minimum), weighted / mass

    def base_fr_intersections(self, joints):
        """Shared plane slack LP, not a distance or penetration-depth estimate."""
        from scipy.optimize import linprog
        poses = self.poses(joints, [0, 0, 0], [1, 0, 0, 0])
        shapes = []
        for i, record in enumerate(self.hull_records):
            if record["owner"] not in ("base", "FR_shank"):
                continue
            rotation, origin = poses[record["owner"]]
            planes = self.hulls[f"planes_{i}"]
            normal = planes[:, :3] @ rotation.T
            offset = planes[:, 3] - normal @ origin
            shapes.append((record, normal, offset))
        candidates = []
        for first, normal_a, offset_a in shapes:
            if first["owner"] != "base":
                continue
            for second, normal_b, offset_b in shapes:
                if second["owner"] != "FR_shank":
                    continue
                normal = np.vstack([normal_a, normal_b])
                rhs = -np.r_[offset_a, offset_b]
                fit = linprog([0, 0, 0, -1], A_ub=np.c_[normal, np.ones(len(normal))],
                              b_ub=rhs, bounds=[(None, None)] * 4, method="highs")
                if not fit.success:
                    raise RuntimeError(fit.message)
                candidates.append({"base_mesh": first["path"], "shank_mesh": second["path"],
                                   "shared_interior_radius_m": float(fit.x[3])})
        return sorted(candidates, key=lambda r: r["shared_interior_radius_m"], reverse=True)[:3]


def trial(geometry, job, directory):
    data = np.load(directory / "telemetry.npz", allow_pickle=False)
    history = json.loads((directory / "status.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    state_index = {int(stamp): i for i, stamp in enumerate(data["state_stamp_ns"])}
    policy = [h for h in history if h["stamp_ns"] in state_index and h["stamp_ns"] % 20_000_000 == 0]
    # Initial sit/rise WBC explicitly assumes four contacts. Only the return
    # from walking supplies the measured mask, so use that phase to infer order.
    paired = [h for h in policy if h["stamp_ns"] >= 17_300_000_000
              and "stance_mask" in h.get("wbc_vectors", {})]
    possible = []
    for order in itertools.permutations(range(4)):
        if all(np.array_equal(np.linalg.norm(data["shank_support_force"][state_index[h["stamp_ns"]]], axis=1)[list(order)] > 5.,
                              h["wbc_vectors"]["stance_mask"]) for h in paired):
            possible.append(order)
    # Diagonal support alone cannot identify swaps within either pair. Use the
    # earlier CAD replay's uniquely identified order and validate it against
    # every measured mask in this recording without claiming uniqueness here.
    mapping_source = ROOT / "reports/v15/usd_20261011/full_suite_20261011/diagnostics/cad_landing_ik_replay.json"
    order = json.loads(mapping_source.read_text())["raw_contact_sensor_to_canonical_indices"]
    assert tuple(order) in possible, (order, possible)
    rows = []
    for h in policy:
        t = h["stamp_ns"] * 1e-9
        if not 16.5 <= t <= 19.04:
            continue
        i = state_index[h["stamp_ns"]]
        points, minimum, com = geometry.measure(data["joint_position"][i], data["position"][i], data["quaternion"][i])
        support = np.linalg.norm(data["shank_support_force"][i], axis=1)[order] > 5.
        assert int(support.sum()) == h["support_feet"]
        mask = h.get("wbc_vectors", {}).get("stance_mask")
        if mask is not None:
            assert np.array_equal(support, mask)
        first, second = points[0, :2], points[3, :2]
        direction = second - first
        distance = float(abs(direction[0] * (com[1] - first[1]) - direction[1] * (com[0] - first[0])) / np.linalg.norm(direction))
        wbc = h["wbc"]
        rows.append(dict(time_s=t, mode=h["mode"], walk_weight=h["walk_weight"],
                         support_count=int(support.sum()), stance_mask=support.tolist(),
                         height_m=float(data["position"][i, 2]),
                         virtual_contact_z_m=points[:, 2].tolist(), authored_foot_min_z_m=minimum.tolist(),
                         com_xy_diagonal_line_distance_m=distance,
                         joint_reference_error_rad=None if "joint_reference" not in h.get("wbc_vectors", {}) else
                         (np.asarray(h["wbc_vectors"]["joint_reference"]) - data["joint_position"][i]).tolist(),
                         wbc_feasible=wbc.get("feasible"), wbc_converged=wbc.get("converged"),
                         max_constraint_violation=wbc.get("max_constraint_violation")))
    subset = [r for r in rows if 18.3 <= r["time_s"] <= 18.8]
    summary = dict(job=job, case="sit_rise_walk_stop_sit", task_success=result["task_success"],
                   telemetry_sha256=digest(directory / "telemetry.npz"),
                   contact_raw_to_canonical_indices=order, contact_permutation_matched_policy_ticks=len(paired),
                   contact_permutation_candidates_this_recording=[list(p) for p in possible],
                   contact_mapping_basis=dict(path=str(mapping_source.relative_to(ROOT)), sha256=digest(mapping_source),
                                              note="Earlier CAD replay identified this order uniquely; this recording validates it but can leave pair swaps ambiguous."),
                   post_blend_window_s=[18.3, 18.8], post_blend_samples=len(subset),
                   post_blend_two_support_samples=sum(r["support_count"] == 2 for r in subset),
                   post_blend_numerically_feasible_samples=sum(r["wbc_feasible"] is True for r in subset),
                   max_nonadjacent_contact_force_n=result["max_nonadjacent_contact_force_n"],
                   snapshots=[r for r in rows if any(abs(r["time_s"] - t) < 1e-8 for t in [17.3, 18., 18.3, 18.5, 18.8, 19.])])
    if not result["task_success"]:
        peak = int(np.argmax(data["nonadjacent_pair_force_n"].max(axis=1)))
        summary["first_self_contact_state_time_s"] = float(data["state_stamp_ns"][np.flatnonzero(data["nonadjacent_pair_force_n"].max(axis=1) > 1.)[0]] * 1e-9)
        summary["contact_pair_peaks_n"] = result["contact_pairs_over_1n"]
        summary["authored_base_fr_hull_screen"] = [dict(state_time_s=float(data["state_stamp_ns"][i] * 1e-9),
            candidates=geometry.base_fr_intersections(data["joint_position"][i])) for i in [peak-1, peak]]
    return summary, rows


def main():
    geometry = RecordedGeometry()
    cases = [(2343, ROOT / "reports/v15/cad_rms_20261011/round2/suite_jobs/2343/results/sit_rise_walk_stop_sit"),
             (2355, ROOT / "reports/v15/jetson_usb_20261012/usb_suite_jobs/2355/results/sit_rise_walk_stop_sit")]
    reports, rows_by_job = [], {}
    for job, directory in cases:
        report, rows = trial(geometry, job, directory)
        reports.append(report)
        rows_by_job[job] = rows
    output = dict(scope="Offline FK replay of recorded Isaac telemetry, with authored CAD hulls; no new physics/controller execution.",
                  nominal_foot_center_vs_original_usd_audit_max_error_m=geometry.nominal_audit_max_error_m,
                  model_sha256=digest(MODEL / "quadruped_cad.xml"), hulls_sha256=digest(MODEL / "collision_hulls.npz"),
                  limitations=["Original baseline/USB runs have different world origins and preceding case histories.",
                               "Authored hull geometry does not reconstruct cooked PhysX hulls or contact offsets.",
                               "A fixed material contact point can differ from the lowest rotated foot-mesh vertex.",
                               "COM distance to a support line is a geometric indicator; it is not a dynamic stability criterion.",
                               "QP feasibility alone does not demonstrate actual physical stability.",
                               "This analysis does not identify a unique causal fix. Controlled comparison2424 remains required."],
                  trials=reports)
    (OUT / "usb2355_handover_geometry.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    with (OUT / "usb2355_handover_geometry.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["job", "time_s", "mode", "walk_weight", "support_count", "height_m", "FR_foot_min_z_m", "RL_foot_min_z_m", "COM_diagonal_distance_m", "wbc_feasible"])
        for job, rows in rows_by_job.items():
            for r in rows:
                writer.writerow([job, r["time_s"], r["mode"], r["walk_weight"], r["support_count"], r["height_m"],
                                 r["authored_foot_min_z_m"][1], r["authored_foot_min_z_m"][2], r["com_xy_diagonal_line_distance_m"], r["wbc_feasible"]])
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, layout="constrained")
    for job, rows in rows_by_job.items():
        t = [r["time_s"] for r in rows]
        label = f"{job}: " + ("original PASS (X=60m)" if job == 2343 else "USB FAIL (X=30m)")
        axes[0].plot(t, [r["height_m"] for r in rows], label=label)
        axes[1].step(t, [r["support_count"] for r in rows], where="post", label=label)
        axes[3].plot(t, [r["com_xy_diagonal_line_distance_m"] * 1000 for r in rows], label=label)
    for leg, index in [("FR", 1), ("RL", 2)]:
        rows = rows_by_job[2355]
        axes[2].plot([r["time_s"] for r in rows], [r["authored_foot_min_z_m"][index] * 1000 for r in rows], label=f"USB {leg} authored foot minimum")
    axes[0].axhline(.2, color="red", linestyle=":", label="failure height")
    axes[2].axhline(0, color="black", linestyle=":")
    for ax, label in zip(axes, ["Root height (m)", "Measured support feet", "Foot minimum Z (mm)", "COM to FL-RR line (mm)"]):
        ax.set_ylabel(label)
        ax.axvline(18.3, color="gray", linestyle="--", label="blend ends")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("Recorded command/state time (s)")
    fig.suptitle("Recorded stop handover: geometry analysis; origin/history differ")
    fig.savefig(OUT / "usb2355_handover_geometry.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"nominal_audit_error_m": geometry.nominal_audit_max_error_m,
                      "trials": [{k: r[k] for k in ["job", "post_blend_samples", "post_blend_two_support_samples", "post_blend_numerically_feasible_samples"]} for r in reports]}, indent=2))


if __name__ == "__main__":
    main()
