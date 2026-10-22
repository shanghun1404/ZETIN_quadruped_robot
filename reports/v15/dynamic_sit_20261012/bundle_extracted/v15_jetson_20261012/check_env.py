#!/usr/bin/env python3
"""Read and record the target environment; never install packages or change the OS."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys


def inspect_environment(os_release=Path("/etc/os-release"), nv_release=Path("/etc/nv_tegra_release"),
                        *, machine=None, environ=None):
    fields = {}
    for line in os_release.read_text().splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            fields[name] = value.strip('"\'')
    architecture = platform.machine() if machine is None else machine
    environment = os.environ if environ is None else environ
    problems = []
    if architecture != "aarch64":
        problems.append(f"aarch64 required; detected {architecture}")
    if fields.get("ID") != "ubuntu" or fields.get("VERSION_ID") != "24.04":
        problems.append(f"Ubuntu 24.04 required; detected {fields.get('ID')} {fields.get('VERSION_ID')}")
    if environment.get("ROS_DISTRO") != "jazzy":
        problems.append(f"ROS_DISTRO=jazzy required after sourcing setup; detected {environment.get('ROS_DISTRO')}")
    bsp = nv_release.read_text().strip() if nv_release.is_file() else None
    return {"architecture": architecture, "os_release": fields, "ros_distro": environment.get("ROS_DISTRO"),
            "python": sys.version, "python_executable": sys.executable,
            "nvidia_bsp_release": bsp, "bsp_source": str(nv_release),
            "cuda_wheel_selection": "unresolved; select against the recorded NVIDIA BSP before GPU deployment",
            "problems": problems}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSON evidence destination")
    args = parser.parse_args()
    try:
        result = inspect_environment()
    except OSError as error:
        parser.exit(1, f"environment inspection failed: {error}\n")
    result["recorded_utc"] = datetime.now(timezone.utc).isoformat()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    sys.exit(1 if result["problems"] else 0)
