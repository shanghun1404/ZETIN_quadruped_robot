import json


def write(root,path,value):
    dest=root/path;dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(value))


def test_missing_runs_stay_incomplete_and_completed_is_not_task_success(tmp_path):
    from summary_report import build_summary
    write(tmp_path,'integration/593/result.json',dict(completed=True,task_success=False,failure=None,seconds_recorded=28.))
    result=build_summary(tmp_path)
    assert result['integration']['593']['status']=='실패'
    assert result['integration']['599']['status']=='미완료'
    assert result['targeted_comparison']['status']=='미완료'
    assert result['display_policy']['label']=='baseline339'
    assert result['adoption_confirmed'] is False


def test_report_uses_existing_screen_selection_without_claiming_adoption(tmp_path):
    from summary_report import build_summary
    metrics=dict(valid=True,survived=True,torque_rms_nm=[1.]*12,torque_peak_nm=[2.]*12)
    write(tmp_path,'comparison/600/comparison_summary.json',dict(mode='targeted_screen',seed=42,
        selected='candidate50',all_gates_pass=True,adoption_pending_heldout=True,
        candidates={'candidate50':dict(status='passed_screen',flat={'candidate':metrics},rough={'candidate':metrics})}))
    result=build_summary(tmp_path)
    assert result['display_policy']['label']=='candidate50'
    assert result['display_policy']['scope']=='seed42 통과 후보 · 채택 미확정'
    assert result['adoption_confirmed'] is False
    assert result['display_policy']['terrains']['flat']['torque_rms_nm']==[1.]*12


def test_reconstructed_same_leg_gap_cannot_clear_measured_crossleg_collision(tmp_path):
    from summary_report import build_summary
    write(tmp_path,'comparison/594/collision_incident.json',dict(max_force_N=218.,selected=False))
    write(tmp_path,'analysis/reconstructed_collision/summary.json',[
        dict(reconstruction_validated=True,minimum_same_leg_thigh_foot_gap_m=.227,
             actual_same_leg_contact_measured=False)])
    result=build_summary(tmp_path)
    assert result['reconstructed_collision'][0]['actual_same_leg_contact_measured'] is False
    assert result['collision_incident']['max_force_N']==218.


def test_future_integration_discovered_and_first_stand_not_mixed_with_return_stand(tmp_path):
    from summary_report import build_summary
    first=dict(mode='STAND',start_s=4.,end_s=7.,duration_s=3.,
               joints=[dict(joint='RL_KFE',torque_rms_nm=17.)])
    second=dict(mode='STAND',start_s=23.,end_s=27.,duration_s=4.,
                joints=[dict(joint='RL_KFE',torque_rms_nm=30.)])
    write(tmp_path,'analysis/integration_reports/integration_601.json',
          dict(task_success=True,completed=True,timeline=[first,second]))
    result=build_summary(tmp_path)
    assert result['latest_integration']=={'job':'601','status':'성공'}
    stand=result['integration']['601']['first_stand']
    assert stand['start_s']==4.
    assert stand['joints'][0]['torque_rms_nm']==17.


def test_standalone_stairs_does_not_replace_latest_full_sequence(tmp_path):
    from summary_report import build_summary
    write(tmp_path,'integration/601/result.json',dict(demo_sequence_required=True,scenario='flat',task_success=False,completed=True))
    write(tmp_path,'integration/602/result.json',dict(demo_sequence_required=False,scenario='stairs_up',task_success=True,completed=True))
    result=build_summary(tmp_path)
    assert result['latest_full_sequence']=={'job':'601','status':'실패'}
    assert result['latest_stairs']=={'job':'602','status':'성공'}
    assert result['integration']['602']['run_kind']=='standalone'
    assert result['latest_integration']['job']=='602'


def test_completed_rejected_screen_records_reason_and_no_adoption(tmp_path):
    from summary_report import build_summary
    gate=dict(all_gates_pass=False,gates={'command_tracking':False,'rms_not_increased':True},mean_joint_rms_ratio=.91,absolute_power_ratio=.84)
    write(tmp_path,'comparison/600/comparison_summary.json',dict(selected=None,all_gates_pass=False,candidates={'candidate100':dict(status='rejected_flat',flat={'gate':gate,'baseline':{'command_speed_mae_m_s':.099277},'candidate':{'command_speed_mae_m_s':.123195}},rough='skipped_after_flat_rejection')}))
    result=build_summary(tmp_path)
    assert result['targeted_comparison']['status']=='완료'
    assert result['training']['598']['evaluation_status']=='후보 전체 탈락'
    assert result['targeted_outcomes']['candidate100']['failed_gates']==['command_tracking']
    assert result['targeted_outcomes']['candidate100']['rough_evaluated'] is False
    assert result['display_policy']['label']=='baseline339'
    assert result['adoption_confirmed'] is False


def test_expected_future_runs_remain_pending_with_separate_scopes(tmp_path):
    from summary_report import build_summary
    write(tmp_path,'training/evaluation_plan.json',{'integration':{'601':{'demo_sequence_required':True,'scenario':'flat'},'602':{'demo_sequence_required':False,'scenario':'stairs_up'}}})
    result=build_summary(tmp_path)
    assert result['latest_full_sequence']=={'job':'601','status':'미완료'}
    assert result['latest_stairs']=={'job':'602','status':'미완료'}


def test_stairs_posthoc_failure_overrides_display_but_preserves_raw_success(tmp_path):
    from summary_report import build_summary
    raw=dict(demo_sequence_required=False,scenario='stairs_up',task_success=True,completed=True)
    write(tmp_path,'integration/602/result.json',raw)
    write(tmp_path,'analysis/stair_602.json',dict(posthoc=True,raw_task_success=True,posthoc_directed_stair_task_success=False,posthoc_full_flight_task_success=False,final_geometry_derived_ground_rise_m=0.))
    result=build_summary(tmp_path)
    assert result['latest_stairs']=={'job':'602','status':'실패'}
    stair=result['integration']['602']
    assert stair['raw_task_success'] is True
    assert stair['task_success'] is False
    assert stair['verdict_source']=='posthoc_directed_and_full_flight'
    assert json.loads((tmp_path/'integration/602/result.json').read_text())==raw
