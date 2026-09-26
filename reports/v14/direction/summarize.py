"""Aggregate fixed-protocol evaluations and plot actual recorded root paths."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
rows, reports = [], []
for path in sorted(ROOT.glob("*_seed*.json")):
    report = json.loads(path.read_text())
    metadata = report["metadata"]
    assert metadata["protocol"] == "fixed_v2"
    assert metadata["effort_guard_fraction"] == .99
    assert metadata["self_collisions_enabled"] and metadata["leg_terrain_filtered"]
    row = dict(
        label=path.stem, seed=metadata["seed"], scenario=metadata["scenario"],
        actor_sha256=metadata["actor_sha256"], heading_control=metadata.get("heading_control", False),
        survived=report["survived_envs"], total=report["initial_envs"],
        directed_stairs=report["path"]["directed_stair_success_count"],
        full_flight=report["path"]["full_flight_success_count"],
        progress_m=report["path"]["mean_final_directed_progress_m"],
        max_cross_track_m=report["path"]["mean_max_abs_cross_track_m"],
        max_heading_deg=report["path"]["mean_max_abs_heading_error_deg"],
        torque_over_limit=report["actuation"]["over_limit_request_fraction"],
        minimum_leg_gap_m=report["interleg_clearance"]["minimum_gap_m"],
        overlap_fraction=report["interleg_clearance"]["negative_gap_frame_fraction"],
    )
    rows.append(row)
    reports.append((path, report))
assert len({r["metadata"]["student_sha256"] for _,r in reports}) <= 1
assert len({r["metadata"]["usd_sha256"] for _,r in reports}) <= 1
(ROOT/"summary.json").write_text(json.dumps(rows, indent=2)+"\n")
for row in rows:
    print(f'{row["label"]}: survive {row["survived"]}/{row["total"]}, directed {row["directed_stairs"]}, full {row["full_flight"]}, progress {row["progress_m"]:.3f}m, cross {row["max_cross_track_m"]:.3f}m, heading {row["max_heading_deg"]:.1f}deg, torque {row["torque_over_limit"]:.3g}')

stairs = [(p,r) for p,r in reports if r["metadata"]["scenario"] == "stairs_up" and r["metadata"]["seed"] == 42 and p.with_suffix(".npz").exists()]
if stairs:
    fig, axes = plt.subplots(1,len(stairs),figsize=(5*len(stairs),5),squeeze=False,sharex=True,sharey=True)
    for ax,(p,r) in zip(axes[0],stairs):
        trace = np.load(p.with_suffix(".npz"))
        for i in range(trace["valid"].shape[1]):
            valid=trace["valid"][:,i]
            pos=trace["positions"][valid,i,:2]-trace["positions"][0,i,:2]
            color="#238b45" if r["path"]["per_environment"][i]["directed_stair_success"] else "#c34b48"
            ax.plot(pos[:,0],pos[:,1],color=color,alpha=.5,lw=1)
        ax.axvspan(-.6,.6,color="#74a9cf",alpha=.13)
        ax.axhline(2.1,color="#333333",ls="--",lw=1)
        ax.set_title(p.stem.replace("_stairs_up_seed42", "")+f'\n{r["path"]["directed_stair_success_count"]}/32 directed passes')
        ax.set(xlabel="Lateral displacement (m)",xlim=(-4,4),ylim=(-4,7))
        ax.grid(alpha=.2)
    axes[0,0].set_ylabel("Forward displacement (m)")
    fig.suptitle("V14: fixed 15.7 cm stairs, first episode only")
    fig.tight_layout()
    fig.savefig(ROOT/"paths_seed42.png",dpi=160)
    plt.close(fig)
