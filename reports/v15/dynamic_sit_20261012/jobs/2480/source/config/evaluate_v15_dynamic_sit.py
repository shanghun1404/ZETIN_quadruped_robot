"""Reuse the authentic CAD/ROS2/USB physical rig for a deeper level crouch.

The new case begins standing and requests sitting at 2s, standing at 8s.
No reset, pose writing or altered torque/collision/fall gates during the trial.
"""
import hashlib
from pathlib import Path

original=Path(__file__).with_name('evaluate_v15_cad_suite.py')
source=original.read_text()


def replace_once(old,new):
    global source
    if source.count(old)!=1:raise RuntimeError('CAD evaluator contract changed: '+old)
    source=source.replace(old,new)


replace_once("stand_names={'standing_hold','sit_rise_walk_stop_sit','deep_crouch_rise_walk_stop_sit'}",
    "stand_names={'standing_hold','sit_rise_walk_stop_sit','deep_crouch_rise_walk_stop_sit','dynamic_sit'}")
replace_once("'v15_cad_suite_controller'", "('v15_dynamic_sit_controller' if name=='dynamic_sit' else 'v15_cad_suite_controller')")
replace_once("'STAND' if name=='standing_hold' else 'SIT'", "'STAND' if name in ('standing_hold','dynamic_sit') else 'SIT'")
replace_once("'false' if name=='standing_hold' else 'true'", "'false' if name in ('standing_hold','dynamic_sit') else 'true'")
replace_once("    env=dict(os.environ,V15_CAD_CASE=name)",
    "    if name=='dynamic_sit':\n        command+=['-p','sit_style:=deep_crouch','-p','posture_transition_s:=2.0']\n    env=dict(os.environ,V15_CAD_CASE=name)")
replace_once("duration=12. if name=='standing_hold' else 30. if standing else spec.duration_s",
    "duration=14. if name=='dynamic_sit' else 12. if name=='standing_hold' else 30. if standing else spec.duration_s")
replace_once("drop=.18 if name=='deep_crouch_rise_walk_stop_sit' else .10",
    "drop=.18 if name=='deep_crouch_rise_walk_stop_sit' else .18 if name=='dynamic_sit' else .10")
replace_once("if name!='standing_hold':", "if name not in ('standing_hold','dynamic_sit'):")
replace_once("drop if standing and name!='standing_hold' else 0.",
    "drop if standing and name not in ('standing_hold','dynamic_sit') else 0.")
replace_once('from v15_cad_suite_metrics import estimated_torque,torque_summary,standing_verdict',
    'from v15_cad_suite_metrics import estimated_torque,torque_summary,standing_verdict\nfrom v15_dynamic_sit_metrics import dynamic_sit_verdict')
replace_once('verdict=standing_verdict(name,arrays,history,completed,nominal_height,nominal_height-drop)',
    "verdict=(dynamic_sit_verdict if name=='dynamic_sit' else standing_verdict)(name,arrays,history,completed,nominal_height,nominal_height-drop)")
replace_once('from v15_cad_ros2_bridge import V15CadBridge',
    'from v15_usb_sim_bridge import UsbCadBridge as V15CadBridge')
replace_once('if bridge is not None:bridge._node.destroy_node()',
    'if bridge is not None:bridge.close_usb();bridge._node.destroy_node()')
print('Dynamic sit rig source SHA256 '+hashlib.sha256(original.read_bytes()).hexdigest(),flush=True)
exec(compile(source,str(original),'exec'),dict(__name__='__main__',__file__=str(original)))
