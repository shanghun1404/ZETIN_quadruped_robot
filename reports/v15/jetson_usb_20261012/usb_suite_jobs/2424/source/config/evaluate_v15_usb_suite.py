"""Run the unchanged physical suite with only the serial codec adapter substituted."""
import hashlib
from pathlib import Path

original=Path(__file__).with_name('evaluate_v15_cad_suite.py')
source=original.read_text()
old_import='from v15_cad_ros2_bridge import V15CadBridge'
old_cleanup='if bridge is not None:bridge._node.destroy_node()'
assert source.count(old_import)==1 and source.count(old_cleanup)==1
patched=source.replace(old_import,'from v15_usb_sim_bridge import UsbCadBridge as V15CadBridge').replace(
    old_cleanup,'if bridge is not None:bridge.close_usb();bridge._node.destroy_node()')
print('USB codec adapter: unchanged criteria source SHA256 '+hashlib.sha256(source.encode()).hexdigest(),flush=True)
exec(compile(patched,str(original),'exec'),dict(__name__='__main__',__file__=str(original)))
