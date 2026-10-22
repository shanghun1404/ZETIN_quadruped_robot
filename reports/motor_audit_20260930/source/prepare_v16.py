"""Generate an isolated measurement copy; original controller source is preserved."""
from pathlib import Path
import hashlib
root = Path(__file__).resolve().parent
source = (root / 'run_v16_original.py').read_text()
def replace(old, new, count=1):
    global source
    assert source.count(old) == count, (old[:80], source.count(old))
    source = source.replace(old, new)
replace('parser.add_argument("--mode",', 'parser.add_argument("--audit-terrain", choices=("flat",), default="flat")\nparser.add_argument("--mode",')
# Audit every physics step, including a command that causes termination.
replace('        self.reset(seed)\n        p =', '        self.reset(seed)\n        motor_samples = []\n        p =')
replace('            tau = self.advance(requested)', '''            audit_valid = alive.clone()
            tau = self.advance(requested)
            motor_samples.append((tau.cpu().numpy().copy(), raw_requested.cpu().numpy().copy(),
                before_vel.cpu().numpy().copy(), audit_valid.cpu().numpy().copy()))''')
replace('        if video:\n            video.close()', '''        if trace:
            np.savez_compressed(self.out / f"motor_trace_seed{seed}.npz",
                applied_torque_nm=np.stack([v[0] for v in motor_samples]),
                computed_torque_nm=np.stack([v[1] for v in motor_samples]),
                joint_velocity_rad_s=np.stack([v[2] for v in motor_samples]),
                alive=np.stack([v[3] for v in motor_samples]),
                time_s=np.arange(len(motor_samples)) * DT, joint_names=np.array(JOINT_NAMES))
        if video:
            video.close()''')
replace('            "base_actuation": False,', '''            "base_actuation": False,
            "audit_terrain": args.audit_terrain,
            "motor_trace_time_convention": "time_s marks 5ms command interval start; velocity before advance; valid before command so terminal command included",
            "audit_note": "Only physics-step telemetry added; frozen 577 controller/parameters; stairs skipped by user request",
            "original_source_sha256": "ORIGINAL_SHA",'''.replace('ORIGINAL_SHA', hashlib.sha256((root/'run_v16_original.py').read_bytes()).hexdigest()))
(root / 'run_v16_physics.py').write_text(source)
compile(source, 'run_v16_physics.py', 'exec')
print('Generated measurement copy, syntax valid')
