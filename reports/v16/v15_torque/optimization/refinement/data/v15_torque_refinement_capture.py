"""V15 guard/LPF experiment using the hash-checked, unchanged baseline recorder."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

baseline_capture = Path(__file__).with_name("v15_torque_capture.py")
baseline_text = baseline_capture.read_text()
baseline_hash = hashlib.sha256(baseline_capture.read_bytes()).hexdigest()
expected_baseline_hash = "042d40629c6a205ac392f47d6880fc79f19ab8d191ae6ae81c118b8b8b8f23de"
if baseline_hash != expected_baseline_hash:
    raise RuntimeError(f"Unreviewed baseline capture: {baseline_hash}")
cutoff_hz = float(os.environ["V15_TORQUE_CUTOFF_HZ"])
label = os.environ["V15_TORQUE_CANDIDATE"]
if cutoff_hz != 20.0:
    raise ValueError("Only the reviewed refinement cutoff 20 Hz is allowed")
# Execute the recorder's construction, then add only the experiment parameter.
prefix, _ = baseline_text.rsplit("exec(compile(source,", 1)
namespace = {"__name__": "v15_candidate_construction", "__file__": str(Path(__file__).resolve())}
exec(compile(prefix, str(baseline_capture), "exec"), namespace)
replace_once = namespace["replace_once"]
replace_once(
    "        use_physx_joint_drives(env_cfg, max_delay_steps=0, guard_fraction=args_cli.effort_guard_fraction)\n",
    "        use_physx_joint_drives(env_cfg, max_delay_steps=0, guard_fraction=args_cli.effort_guard_fraction)\n"
    + f"        env_cfg.actions.joint_pos.cutoff_hz = {cutoff_hz!r}\n",
)
replace_once(
    '                "actual_solver_motor_torque_recorded": False,\n',
    '                "actual_solver_motor_torque_recorded": False,\n'
    + f'                "candidate_label": {label!r},\n'
    + f'                "requested_cutoff_hz": {cutoff_hz!r},\n'
    + '                "actual_cutoff_hz": float(audit_action.cfg.cutoff_hz),\n'
    + '                "actual_guard_fraction": float(audit_action.cfg.effort_guard_fraction),\n'
    + '                "actual_lpf_alpha": float(audit_action._alpha),\n'
    + f'                "baseline_capture_sha256": {baseline_hash!r},\n',
)
exec(compile(namespace["source"], str(namespace["ORIGINAL"]), "exec"),
     {"__name__": "__main__", "__file__": str(Path(__file__).resolve())})
