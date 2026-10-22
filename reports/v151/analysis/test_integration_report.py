import importlib.util
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest


def report_module():
    spec=importlib.util.spec_from_file_location('integration_report',Path(__file__).with_name('integration_report.py'))
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def fixture(tmp_path, *, bad=False):
    names=[f'{leg}_{j}' for leg in ('FL','FR','RL','RR') for j in ('HAA','HFE','KFE')][::-1]
    tau=np.zeros((4,12));tau[:,-1]=[10,0,10,0]
    if bad: tau[2,-1]=np.nan
    np.savez(tmp_path/'result.npz',estimated_torque=tau,joint_velocity=np.zeros((4,12)),
        position=np.tile([0,0,.464],(4,1)),velocity=np.zeros((4,3)),quaternion=np.tile([1,0,0,0],(4,1)),
        gaps=np.full((4,24),.03),trunk_gaps=np.full((4,8),-.005),contact_pair_force=np.zeros((4,12)))
    (tmp_path/'result.json').write_text(json.dumps(dict(joint_order=names,completed=True,task_success=False,failure=None)))
    status=[dict(stamp_ns=i*5000000,mode=m,compute_ms=1,support_feet=4,wbc={}) for i,m in enumerate(['SIT','WALK','SIT','SIT'])]
    (tmp_path/'result.status.json').write_text(json.dumps(status))
    return tmp_path


def test_mode_statistics_keep_canonical_order_failure_and_disjoint_durations(tmp_path):
    r=report_module().summarize(fixture(tmp_path))
    assert r['task_success'] is False
    assert r['modes']['SIT']['duration_s']==pytest.approx(.015)
    j=r['modes']['SIT']['joints'][0]
    assert j['joint']=='FL_HAA'
    assert j['torque_rms_nm']==pytest.approx(np.sqrt(200/3))
    assert j['rated_exceed_fraction']==pytest.approx(2/3)
    assert j['rated_exceed_s']==pytest.approx(.01)
    assert j['rated_exceed_longest_s']==pytest.approx(.005)
    assert j['ideal_current_rms_a']==pytest.approx(np.sqrt(200/3)/1.23)
    assert r['modes']['SIT']['collision']['trunk_proxy_overlap_fraction']==1
    assert r['modes']['SIT']['collision']['crossleg_contact_over_1n_fraction']==0


def test_first_nonfinite_sample_excludes_later_samples(tmp_path):
    r=report_module().summarize(fixture(tmp_path,bad=True))
    assert r['samples_used']==2
    assert r['data_valid'] is False
    assert r['task_success'] is False


def test_drive_budget_violation_is_not_passed(tmp_path):
    root=fixture(tmp_path);path=root/'result.npz';a=dict(np.load(path));a.update(drive_kp=np.zeros((4,12)),drive_kd=np.zeros((4,12)),feedforward=np.ones((4,12)),drive_limit=np.full((4,12),48.));np.savez(path,**a)
    r=report_module().summarize(root)
    assert r['actuation_checks']['drive_budget_arrays']['passed'] is False
    assert r['actuation_checks']['drive_gain_getters']['available'] is False


@pytest.mark.parametrize('extra, expected', [(0., True), (.2, False)])
def test_physx_cap_getter_checks_policy_order_and_separate_effort(tmp_path,extra,expected):
    root=fixture(tmp_path);module=report_module()
    limits=np.array([24.8 if name.endswith('HAA') else 48. for name in module.POLICY])
    ff=np.arange(12,dtype=float)/4
    actual=limits-ff;actual[0]+=extra
    path=root/'result.json';meta=json.loads(path.read_text())
    meta['drive_limit_checks']=[dict(stamp_ns=0,drive_limit=actual.tolist(),feedforward=ff.tolist())]
    path.write_text(json.dumps(meta))
    check=module.summarize(root)['actuation_checks']['drive_limit_getters']
    assert check['passed'] is expected
    assert check['max_budget_excess_nm']==pytest.approx(extra)


def test_report_retains_shift_and_separates_same_leg_proxy_from_actual_contacts(tmp_path):
    root=fixture(tmp_path);path=root/'result.npz';arrays=dict(np.load(path))
    arrays['same_leg_thigh_foot_gap_m']=np.array([[-.01,.02,.03,.04],[.01,.02,.03,.04],[.02,.03,.04,.05],[.01,.02,-.02,.04]])
    np.savez(path,**arrays)
    path=root/'result.status.json';statuses=json.loads(path.read_text())
    for s in statuses:s['stand_forward_shift_m']=.03
    path.write_text(json.dumps(statuses))
    path=root/'result.json';metadata=json.loads(path.read_text())
    pairs=['FL_thigh_profile--FL_foot','FR_thigh_profile--FR_foot','RL_thigh_profile--RL_foot','RR_thigh_profile--RR_foot']
    metadata.update(same_leg_thigh_foot_pair_order=pairs,same_leg_thigh_foot_geometry={'FL':{'proxy':'enclosing box'}},same_leg_thigh_foot_gate_applied=False,min_same_leg_thigh_foot_gap_m=-.02)
    path.write_text(json.dumps(metadata))
    module=report_module();r=module.write_report(root,tmp_path/'output')
    assert r['stand_forward_shift_m']==pytest.approx(.03)
    proxy=r['modes']['SIT']['same_leg_thigh_foot_proxy']
    assert proxy['available']
    assert proxy['minimum_gap_m']==pytest.approx(-.02)
    assert proxy['overlap_fraction']==pytest.approx(2/3)
    assert proxy['per_pair_minimum_m'][pairs[0]]==pytest.approx(-.01)
    assert proxy['actual_contact_measured'] is False
    assert r['modes']['SIT']['collision']['crossleg_contact_over_1n_fraction']==0
    assert r['same_leg_thigh_foot_evidence']['geometry']==metadata['same_leg_thigh_foot_geometry']
    text=(tmp_path/'output'/f'integration_{root.name}.html').read_text()
    assert '+0.030 m' in text
    assert '같은 다리 허벅지–발 근사' in text
    assert '실제 접촉 측정이 아닙니다' in text


def test_missing_shift_and_same_leg_proxy_are_not_assumed_zero_or_safe(tmp_path):
    r=report_module().summarize(fixture(tmp_path))
    assert r['stand_forward_shift_m'] is None
    assert r['stand_forward_shift_recorded_fraction']==0.
    p=r['modes']['SIT']['same_leg_thigh_foot_proxy']
    assert p['available'] is False
    assert p['minimum_gap_m'] is None
    assert p['overlap_fraction'] is None


def test_contiguous_stand_segments_retain_distinct_torque_motion_and_collision(tmp_path):
    root=fixture(tmp_path);path=root/'result.status.json';statuses=json.loads(path.read_text())
    for s in statuses:
        if s['mode']=='SIT':s['mode']='STAND'
    path.write_text(json.dumps(statuses))
    path=root/'result.npz';a=dict(np.load(path));a['estimated_torque'][2,-1]=20.
    a['position'][2:,2]=[.45,.44];a['velocity'][2:,1]=[.2,.4]
    a['contact_pair_force'][2,0]=2.;a['gaps'][2,0]=-.01
    np.savez(path,**a)
    module=report_module();r=module.write_report(root,tmp_path/'output')
    first,walk,second=r['timeline']
    assert first['mode']=='STAND' and second['mode']=='STAND'
    assert first['mode_occurrence']==1 and second['mode_occurrence']==2
    assert second['segment_index']==3
    assert first['joints'][0]['torque_rms_nm']==pytest.approx(10.)
    assert second['joints'][0]['torque_rms_nm']==pytest.approx(np.sqrt(200.))
    assert second['joints'][0]['torque_peak_nm']==pytest.approx(20.)
    assert second['joints'][0]['rated_exceed_fraction']==pytest.approx(.5)
    assert second['joints'][0]['rated_exceed_longest_s']==pytest.approx(.005)
    assert second['height_min_m']==pytest.approx(.44)
    assert second['speed_mean_m_s']==pytest.approx(.3)
    assert second['speed_max_m_s']==pytest.approx(.4)
    assert first['collision']['maximum_crossleg_contact_force_n']==0
    assert second['collision']['maximum_crossleg_contact_force_n']==2.
    assert second['collision']['crossleg_capsule_overlap_fraction']==pytest.approx(.5)
    assert sum(v['duration_s'] for v in r['timeline'])==pytest.approx(r['seconds_used'])
    with (tmp_path/'output'/f'integration_{root.name}_segments.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==3 and rows[2]['mode_occurrence']=='2'
    assert float(rows[2]['height_min_m'])==pytest.approx(.44)
    with (tmp_path/'output'/f'integration_{root.name}_segment_joints.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==36
    assert float(rows[24]['torque_rms_nm'])==pytest.approx(np.sqrt(200.))
    page=(tmp_path/'output'/f'integration_{root.name}.html').read_text()
    assert 'STAND #1' in page and 'STAND #2' in page


def stair_fixture(tmp_path):
    root=tmp_path/'integration'/'602';root.mkdir(parents=True);fixture(root)
    path=root/'result.json';meta=json.loads(path.read_text());meta.update(scenario='stairs_up',task_success=True)
    path.write_text(json.dumps(meta))
    return root


def test_stair_survival_is_not_ascent_without_posthoc_evidence(tmp_path):
    root=stair_fixture(tmp_path);module=report_module();r=module.write_report(root,tmp_path/'out')
    assert r['raw_task_success'] is True
    assert r['task_success'] is None
    assert r['task_verdict']=='unverified'
    assert r['stair_assessment']['available'] is False
    page=(tmp_path/'out'/'integration_602.html').read_text()
    assert '전체 과제 미검증' in page


@pytest.mark.parametrize('stale',[False,True])
def test_stair_posthoc_overrides_survival_with_matching_source_provenance(tmp_path,stale):
    root=stair_fixture(tmp_path);analysis=tmp_path/'analysis';analysis.mkdir()
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    evidence=dict(posthoc=True,posthoc_directed_stair_task_success=False,posthoc_full_flight_task_success=False,
        final_geometry_derived_ground_rise_m=0.,limitations=['ground reconstructed from mesh'],
        input_sha256={str(root/name):digest for name,digest in before.items() if name in ('result.json','result.npz')})
    if stale:evidence['input_sha256'][str(root/'result.npz')]='0'*64
    path=analysis/'stair_602.json';path.write_text(json.dumps(evidence))
    r=report_module().write_report(root,tmp_path/'out')
    assert r['raw_task_success'] is True
    assert r['task_success'] is (None if stale else False)
    assert r['task_verdict']==('unverified' if stale else 'failed')
    assert r['stair_assessment']['source']==str(path.resolve())
    assert r['stair_assessment']['source_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert r['stair_assessment']['provenance_matches'] is (not stale)
    assert 'ground reconstructed from mesh' in r['stair_assessment']['limitations']
    assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}==before
    page=(tmp_path/'out'/'integration_602.html').read_text()
    assert ('전체 과제 미검증' if stale else '전체 과제 실패') in page
    assert '사후 계단 판정' in page


@pytest.mark.parametrize('completed',[True,False])
def test_valid_posthoc_ascent_uses_completion_not_legacy_raw_success_flag(tmp_path,completed):
    root=stair_fixture(tmp_path);path=root/'result.json';meta=json.loads(path.read_text())
    meta.update(task_success=False,completed=completed);path.write_text(json.dumps(meta))
    analysis=tmp_path/'analysis';analysis.mkdir()
    evidence=dict(posthoc_directed_stair_task_success=True,posthoc_full_flight_task_success=True,
        input_sha256={str(root/name):hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('result.json','result.npz')})
    (analysis/'stair_602.json').write_text(json.dumps(evidence))
    r=report_module().summarize(root)
    assert r['raw_task_success'] is False
    assert r['task_success'] is completed
