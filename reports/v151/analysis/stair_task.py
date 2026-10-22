"""Posthoc stair verdict using existing V14 path criteria and reconstructed terrain.

For the V15.1 fixed level-5 inverted-pyramid run only. This adds a separate
interpretation of raw completion; it never changes the simulation artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'config'))
from v14_path_metrics import summarize_path


def ground_rise_from_center(relative_xy,step_height):
    """Known square pit: 8m tile, 1m border, 0.3m tread, six rings.

    Reproduce installed IsaacLab mesh_terrains.inverted_pyramid_stairs_terrain:
    inner terrain half-width=3m; six rings leave a 1.2m center half-width.
    The bottom is seven risers below the outer border. At a shared vertical
    boundary use the higher adjacent horizontal face; this is geometric
    sampling, not a measured ray or contact height.
    """
    xy=np.asarray(relative_xy,dtype=float)
    if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all():
        raise ValueError('Expected finite [sample,2] XY positions')
    if not np.isfinite(step_height) or step_height<=0:
        raise ValueError('step height must be positive')
    radius=np.abs(xy).max(axis=1)
    depth_steps=np.clip(np.ceil((3.-radius)/.3-1e-10),0,7)
    return (7-depth_steps)*step_height


def analyze(arrays,raw):
    if raw.get('scenario')!='stairs_up':
        raise ValueError('Only the fixed stairs_up geometry is supported')
    geometry=raw.get('geometry',{})
    height=geometry.get('step_height_m',float('nan'))
    rise=geometry.get('full_rise_m',float('nan'))
    if not np.isclose(height,.05+(.20-.05)*5/7,atol=1e-10,rtol=0) or not np.isclose(rise,1.1,atol=1e-10,rtol=0):
        raise ValueError('Unknown geometry: expected fixed level 5, seven risers totaling 1.1m')
    position=np.asarray(arrays['position'],float)
    quat=np.asarray(arrays['quaternion'],float)
    if position.ndim!=2 or position.shape[1]!=3 or not len(position) or quat.shape!=(len(position),4):
        raise ValueError('Expected synchronized nonempty root pose arrays')
    if not np.isfinite(position).all() or not np.isfinite(quat).all():
        raise ValueError('Root poses must be finite')
    if np.max(np.abs(np.linalg.norm(quat,axis=1)-1))>1e-5:
        raise ValueError('Root quaternions must be unit length')
    # Initial state pos=(0,0,z) and reset XY offsets=(0,0): sample zero is
    # the tile center. This is an explicit protocol assumption, not telemetry.
    relative_xy=position[:,:2]-position[0,:2]
    if np.max(np.abs(relative_xy))>24.:
        raise ValueError('Trajectory leaves the known 8m tile plus 20m outer border')
    ground=ground_rise_from_center(relative_xy,height)
    survived=bool(raw.get('completed') and raw.get('failure') is None)
    path=summarize_path(positions=position[:,None],quaternions=quat[:,None],
        ground_height=ground[:,None],valid=np.ones((len(position),1),bool),
        alive_at_end=np.array([survived]),scenario='stairs_up',step_height_m=height,full_rise_m=rise)
    for name,width in [('gaps',24),('same_leg_thigh_foot_gap_m',4),('contact_pair_force',12)]:
        value=np.asarray(arrays[name])
        if value.shape!=(len(position),width) or not np.isfinite(value).all():
            raise ValueError(f'Invalid collision channel {name}')
    collision=dict(cross_leg_force_max_N=float(np.max(arrays['contact_pair_force'])),
                   cross_leg_capsule_minimum_m=float(np.min(arrays['gaps'])),
                   same_leg_foot_box_minimum_m=float(np.min(arrays['same_leg_thigh_foot_gap_m'])))
    collision_pass=bool(collision['cross_leg_force_max_N']<=1.
        and collision['cross_leg_capsule_minimum_m']>=-.002 and collision['same_leg_foot_box_minimum_m']>=0.)
    row=path['per_environment'][0]
    return dict(schema=1,posthoc=True,raw_verdict_modified=False,
        raw_task_success=raw.get('task_success'),raw_motion_success=raw.get('motion_success'),
        raw_collision_checks_pass=raw.get('collision_checks_pass'),raw_completed=raw.get('completed'),
        raw_failure=raw.get('failure'),
        raw_success_interpretation='The raw non-demo task_success flag means completion plus collision checks; it does not establish stair ascent.',
        success_interpretation_override='Use the separate posthoc directed-stair/full-flight verdicts below to assess ascent; original raw fields are preserved.',
        posthoc_directed_stair_task_success=bool(row['directed_stair_success'] and collision_pass),
        posthoc_full_flight_task_success=bool(row['full_flight_success'] and collision_pass),
        collision_checks_pass=collision_pass,collision=collision,path=path,
        step_height_m=float(height),full_rise_m=float(rise),
        terrain_center_xy_inferred_from_first_root_sample_m=position[0,:2].tolist(),
        center_provenance='guide_robot_cfg init_state XY=(0,0); evaluator reset XY range=(0,0); first recorded root pose precedes first physics step',
        ground_height_source='analytically reconstructed inverted-pyramid terrain under recorded root XY; not measured ray/contact heights',
        final_geometry_derived_ground_rise_m=float(ground[-1]),
        maximum_geometry_derived_ground_rise_m=float(ground.max()),
        final_root_height_change_m=float(position[-1,2]-position[0,2]),
        terrain_parameters=dict(size_m=[8.,8.],border_width_m=1.,step_width_m=.3,
                                requested_platform_width_m=3.,generated_center_half_width_m=1.2,rings=6),
        limitations=[
            'Posthoc analysis; thresholds reuse existing v14_path_metrics.py but were not applied by the raw ROS2 evaluator.',
            'Ground is reconstructed from the configured single-tile mesh and initial center assumption; actual terrain origins and ground rays were not recorded.',
            'The geometry-derived ground height is under the root XY, not beneath each foot; this is not a measurement of foot support.',
            'Same-leg negative gaps mean conservative foot-box proxy overlap, not confirmed physical mesh contact.',
            'No new simulation, original result change, or claim beyond the existing directed/full-flight corridor contract.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.result.with_suffix('.npz')
    if args.output.resolve() in (args.result.resolve(),source.resolve()):
        parser.error('Output must be separate from raw simulation artifacts')
    arrays=dict(np.load(source,allow_pickle=False))
    result=analyze(arrays,json.loads(args.result.read_text()))
    sources=[args.result,source,Path(__file__),ROOT/'config/v14_path_metrics.py',
             ROOT/'config/v14_eval_protocol.py',ROOT/'config/v9_env_cfg.py',ROOT/'config/guide_robot_cfg.py']
    result['input_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    result['installed_mesh_source']='/opt/isaaclab/source/isaaclab/isaaclab/terrains/trimesh/mesh_terrains.py:inverted_pyramid_stairs_terrain'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('posthoc_directed_stair_task_success','posthoc_full_flight_task_success','final_geometry_derived_ground_rise_m')},indent=2))


if __name__=='__main__':
    main()
