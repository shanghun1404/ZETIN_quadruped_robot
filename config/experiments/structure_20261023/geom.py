"""Geometry helpers: STL I/O, USD placements of quadruped parts (FL leg + body)."""
import os, re, json, struct
from pathlib import Path
import numpy as np

# Extracted quadruped_robot_20261023.zip (without gcode/3mf); see reports/structure_20261023/README.md §8.
ROOT = Path(os.environ.get('QR_ROOT', Path(__file__).resolve().parent.parent / 'qr20261023' / 'quadruped_robot'))
USD = ROOT / 'isaac_usd' / 'quadruped_robot.usda'
PRINT = ROOT / 'print_stl'
MESHES = ROOT / 'isaac_usd' / 'meshes'


def read_stl(path):
    data = Path(path).read_bytes()
    if data[:5] == b'solid' and b'facet' in data[:400]:
        nums = re.findall(rb'vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)', data)
        tri = np.array(nums, dtype=float).reshape(-1, 3, 3)
    else:
        n = struct.unpack('<I', data[80:84])[0]
        rec = np.frombuffer(data, dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), count=n, offset=84)
        tri = rec['v'].astype(float)
    flat = tri.reshape(-1, 3)
    key = np.round(flat, 6)
    V, inv = np.unique(key, axis=0, return_inverse=True)
    F = inv.reshape(-1, 3)
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    return V, F


def sanitize(name):
    return re.sub(r'[^A-Za-z0-9_]', '_', name)


def usd_placements(path=USD):
    """{link: [(component, material, 4x4 row-vector matrix)]} from the robot USD."""
    text = Path(path).read_text()
    out = {}
    for lm in re.finditer(r'def Xform "(\w+)" \(\s*prepend apiSchemas = \["PhysicsRigidBodyAPI"', text):
        start = lm.end()
        depth = 0; i = text.index('{', start); j = i
        while True:
            c = text[j]
            if c == '{': depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0: break
            j += 1
        body = text[i:j]
        items = []
        for m in re.finditer(r'</Meshes/([^>]+)>\s*\)\s*\{(.*?)\n\s*\}', body, re.S):
            blk = m.group(2)
            mat = re.search(r'physics_materials/(\w+)>', blk)
            tr = re.search(r'matrix4d xformOp:transform = \((.*?)\)\s*\n', blk)
            if not tr: continue
            M = np.array([[float(v) for v in r.split(',')] for r in re.findall(r'\(([^()]*)\)', tr.group(1))])
            items.append((m.group(1), mat.group(1) if mat else None, M))
        out[lm.group(1)] = items
    return out


def transform(V_m, M):
    """Row-vector USD transform of points in metres."""
    return (np.c_[V_m, np.ones(len(V_m))] @ M)[:, :3]
