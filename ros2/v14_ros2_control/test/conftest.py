import sys
from pathlib import Path

# Nodes import the repository's flat config/ modules (v13_torque_mdp, v13_depth_policy, ...).
_REPOSITORY = Path(__file__).resolve().parents[3]
for _path in (_REPOSITORY / "config", _REPOSITORY / "ros2" / "v14_ros2_control"):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
