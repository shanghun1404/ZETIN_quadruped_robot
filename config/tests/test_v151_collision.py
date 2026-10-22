"""Geometric capsule/trunk checks independent of PhysX collider activation."""

import numpy as np
import pytest


def test_segment_box_minimum_can_be_inside_segment_near_face_or_corner():
    from v151_collision import segment_aabb_distance_m

    starts = np.array([[-2., 1.2, 0.], [1.5, 2., 0.], [2., 2., 2.], [-2., 0., 0.]])
    ends = np.array([[2., 1.2, 0.], [2., 1.5, 0.], [2., 2., 2.], [2., 0., 0.]])
    np.testing.assert_allclose(segment_aabb_distance_m(starts, ends, [1., 1., 1.]),
                               [.2, 1.0606601717798212, 1.7320508075688772, 0.], atol=1e-14)


def poses():
    names = ["base"] + [f"{leg}_{part}_profile" for leg in ("FL", "FR", "RL", "RR")
                        for part in ("thigh", "calf")]
    positions = np.zeros((1, 9, 3))
    quats = np.zeros((1, 9, 4))
    quats[..., 0] = 1.
    return names, positions, quats


def test_trunk_gap_uses_actor_frame_box_and_capsule_offsets():
    from v151_collision import capsule_trunk_gaps_m

    names, positions, quats = poses()
    # Capsule z=.24; trunk top=.15; radius=.02 -> gap=.07.
    np.testing.assert_allclose(capsule_trunk_gaps_m(positions, quats, names), .07)
    # Minimum centerline (x,z)=(.32,.24); nearest trunk corner=(.2,.15).
    positions[0, 1, 0] = .31
    assert capsule_trunk_gaps_m(positions, quats, names)[0, 0] == pytest.approx(.13)


def test_world_rotation_translation_and_body_reordering_preserve_gaps():
    from v151_collision import capsule_trunk_gaps_m

    names, positions, quats = poses()
    positions[0, 1, 0] = .31
    expected = capsule_trunk_gaps_m(positions, quats, names)
    # Common 90-degree Y rotation plus translation; trunk and profiles rotate.
    rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    transformed = positions @ rotation.T + [3., -2., .4]
    quats[:] = [2 ** -.5, 0., 2 ** -.5, 0.]
    order = np.arange(8, -1, -1)
    actual = capsule_trunk_gaps_m(transformed[:, order], quats[:, order], [names[i] for i in order])
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_profile_rotation_changes_capsule_axis_in_world():
    from v151_collision import capsule_trunk_gaps_m

    names, positions, quats = poses()
    # Rotate first profile about Y so centerline spans z=-.01 through -.24,
    # at x=.24; trunk extends x<=.2 and z>=0, so closest distance=sqrt(.0017).
    quats[0, 1] = [2 ** -.5, 0., 2 ** -.5, 0.]
    assert capsule_trunk_gaps_m(positions, quats, names)[0, 0] == pytest.approx(.0212310562561766)


def test_overlap_summary_excludes_dead_frames_and_does_not_claim_penetration_depth():
    from v151_collision import capsule_trunk_gaps_m, summarize_trunk_clearance

    names, positions, quats = poses()
    positions[0, 1, 2] = -.2  # centerline intersects box, gap saturates at -radius.
    gaps = capsule_trunk_gaps_m(positions, quats, names)
    assert gaps[0, 0] == pytest.approx(-.02)
    samples = np.stack([gaps, np.full_like(gaps, -1.)])
    summary = summarize_trunk_clearance(samples, np.array([[True], [False]]))
    assert summary["minimum_gap_m"] == pytest.approx(-.02)
    assert summary["negative_gap_frame_fraction"] == 1.
    assert summary["minimum_gap_profile"] == "FL_thigh_profile"
    assert summary["penetration_depth_measured"] is False
    assert summary["actual_contact_measured"] is False


def test_invalid_geometry_fails_instead_of_reporting_clearance():
    from v151_collision import capsule_trunk_gaps_m, segment_aabb_distance_m

    with pytest.raises(ValueError):
        segment_aabb_distance_m([0., 0., 0.], [1., 0., 0.], [1., -1., 1.])
    names, positions, quats = poses()
    quats[0, 0] = 0.
    with pytest.raises(ValueError):
        capsule_trunk_gaps_m(positions, quats, names)


def test_foot_mesh_bound_includes_composed_scale_rotation_and_offset():
    from v151_collision import foot_mesh_box_from_points

    # USD uses row-vector transforms. Local box center=(2,3,4), scaled 2,
    # rotated 90deg about Z, then translated (10,20,30).
    vertices = np.array([[x,y,z] for x in (1.,3.) for y in (1.,5.) for z in (3.,5.)])
    matrix = np.array([[0.,2.,0.,0.],[-2.,0.,0.,0.],[0.,0.,2.,0.],[10.,20.,30.,1.]])
    box = foot_mesh_box_from_points(vertices, matrix)
    np.testing.assert_allclose(box['center_local_m'], [4.,24.,38.])
    np.testing.assert_allclose(box['half_extents_m'], [4.,2.,2.])
    transformed = vertices @ matrix[:3,:3] + matrix[3,:3]
    assert np.all(np.abs(transformed - box['center_local_m']) <= box['half_extents_m'])


def foot_poses():
    names = [f'{leg}_{part}' for leg in ('FL','FR','RL','RR') for part in ('thigh_profile','foot')]
    pos = np.zeros((1,8,3)); quat = np.zeros((1,8,4)); quat[...,0] = 1.
    boxes = [dict(center_local_m=[.125,.09,.24], half_extents_m=[.02,.02,.02]) for _ in range(4)]
    return names,pos,quat,boxes


def test_same_leg_thigh_foot_gap_resolves_offsets_and_pair_order():
    from v151_collision import same_leg_thigh_foot_gaps_m

    names,pos,quat,boxes = foot_poses()
    # Parallel body frames: z separation .10 minus box .02 and capsule .02.
    pos[:,1::2,2] = [.10,.20,.30,.40]
    np.testing.assert_allclose(same_leg_thigh_foot_gaps_m(pos,quat,names,boxes), [[.06,.16,.26,.36]])
    pos[0,1,2] = 0.
    assert same_leg_thigh_foot_gaps_m(pos,quat,names,boxes)[0,0] == pytest.approx(-.02)


def test_same_leg_clearance_is_invariant_to_common_pose_and_body_order():
    from v151_collision import same_leg_thigh_foot_gaps_m

    names,pos,quat,boxes = foot_poses(); pos[:,1::2,2] = [.10,.20,.30,.40]
    # Rotate all frames 90deg Y; each box must rotate with its own foot body.
    rot = np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    pos = pos @ rot.T + [2.,-3.,5.]; quat[:] = [2**-.5,0.,2**-.5,0.]
    order = np.arange(7,-1,-1)
    actual = same_leg_thigh_foot_gaps_m(pos[:,order],quat[:,order],[names[i] for i in order],boxes)
    np.testing.assert_allclose(actual, [[.06,.16,.26,.36]], atol=1e-14)


def test_rotating_only_the_foot_rotates_its_offset_box():
    from v151_collision import same_leg_thigh_foot_gaps_m

    names,pos,quat,boxes = foot_poses()
    quat[0,1] = [2**-.5,2**-.5,0.,0.]
    # Foot box center rotates to (.125,-.24,.09). Nearest box face/corner
    # to the thigh centerline has y/z residuals .31/.13 before capsule radius.
    gap = same_leg_thigh_foot_gaps_m(pos,quat,names,boxes)[0,0]
    assert gap == pytest.approx(np.sqrt(.31**2+.13**2)-.02)


def test_foot_proxy_rejects_invalid_geometry_instead_of_claiming_clearance():
    from v151_collision import foot_mesh_box_from_points, same_leg_thigh_foot_gaps_m

    with pytest.raises(ValueError): foot_mesh_box_from_points([[0.,0.,np.nan]],np.eye(4))
    with pytest.raises(ValueError): foot_mesh_box_from_points([[0.,0.,0.]],np.eye(4))
    names,pos,quat,boxes = foot_poses(); boxes[0]['half_extents_m'][0] = -1.
    with pytest.raises(ValueError): same_leg_thigh_foot_gaps_m(pos,quat,names,boxes)
