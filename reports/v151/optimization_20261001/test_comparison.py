"""Synthetic evidence tests: these catch unsafe acceptance, not report wording."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
import numpy as np

P=Path(__file__).with_name('comparison.py')
if P.exists():
    spec=importlib.util.spec_from_file_location('comparison',P)
    comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)
else:
    comparison=None


def fixture():
    n=5600;t=np.arange(n)*.005
    names=[f'{leg}_{joint}' for leg in ('FL','FR','RL','RR') for joint in ('HAA','HFE','KFE')]
    stages=[(0,'SIT'),(.5,'RISE'),(3.5,'STABILIZE'),(3.92,'STAND'),(7,'BLEND_WALK'),(8,'WALK'),(17,'BRAKE'),(17.3,'BLEND_STAND'),(18.3,'STABILIZE'),(18.8,'STAND'),(23,'LOWER'),(26,'SETTLE_SIT'),(26.02,'SIT')]
    modes=np.full(n,'SIT',dtype='<U20')
    for start,mode in stages:modes[t>=start]=mode
    tau=np.tile([2.,4.,10.],(n,4))
    statuses=[dict(stamp_ns=i*5_000_000,mode=str(modes[i]),command=[0.,.4 if modes[i]=='WALK' else 0.,0.],bounded_torque=tau[i].tolist()) for i in range(n)]
    arrays=dict(estimated_torque=tau,joint_velocity=np.ones((n,12)),position=np.column_stack([t*0,t*.15,t*0+.364]),velocity=np.column_stack([t*0,np.where(modes=='WALK',.4,0),t*0]),quaternion=np.tile([1.,0,0,0],(n,1)),gaps=np.ones((n,24))*.1,trunk_gaps=np.ones((n,8))*.1,same_leg_thigh_foot_gap_m=np.ones((n,4))*.1,contact_pair_force=np.zeros((n,12)))
    metadata=dict(scenario='flat',seed=42,seconds_recorded=28.,completed=True,task_success=True,motion_success=True,collision_checks_pass=True,reset_after_initialization=False,failure=None,demo_sequence_required=True,joint_order=names,collision_scope='synthetic',collision_gate_semantics='synthetic',same_leg_thigh_foot_gate_applied=True)
    return arrays,metadata,statuses


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(comparison,'comparison implementation must exist')
        self.base=fixture();self.candidate=copy.deepcopy(self.base)
        self.candidate[0]['estimated_torque']*=.8
        for row in self.candidate[2]:row['bounded_torque']=(np.array(row['bounded_torque'])*.8).tolist()
    def run_comparison(self):
        return comparison.compare(comparison.analyze(*self.base),comparison.analyze(*self.candidate))
    def test_lower_torque_same_motion_accepted(self):
        self.assertTrue(self.run_comparison()['accepted'])
    def test_missing_poststop_stand_rejected(self):
        for row in self.candidate[2]:
            if 18.8e9<=row['stamp_ns']<23e9:row['mode']='STABILIZE'
        self.assertFalse(self.run_comparison()['gates']['complete_motion_sequence']['passed'])
    def test_overspeed_rejected_even_with_lower_torque(self):
        self.candidate[0]['velocity'][1600:3400,1]=.8
        self.assertFalse(self.run_comparison()['gates']['walk_command_mae']['passed'])
    def test_collision_rejected_even_when_metadata_claims_pass(self):
        self.candidate[0]['contact_pair_force'][10,0]=1.1
        self.assertFalse(self.run_comparison()['gates']['collision']['passed'])
    def test_late_final_sit_rejected(self):
        for row in self.candidate[2]:
            if 26.5e9<=row['stamp_ns']<27e9:row['mode']='SETTLE_SIT'
        self.assertFalse(self.run_comparison()['gates']['final_sit_window']['passed'])
    def test_poststop_torque_increase_not_hidden_by_other_modes(self):
        self.candidate[0]['estimated_torque'][3760:4600]*=2
        self.assertFalse(self.run_comparison()['gates']['poststop_stand_rms']['passed'])
    def test_nonfinite_not_discarded_into_success(self):
        self.candidate[0]['estimated_torque'][-1,0]=np.nan
        self.assertFalse(self.run_comparison()['gates']['finite_no_reset']['passed'])
    def test_stamp_gap_rejected(self):
        self.candidate[2][100]['stamp_ns']+=5_000_000
        with self.assertRaises(ValueError):comparison.analyze(*self.candidate)
    def test_short_run_rejected(self):
        self.candidate=(dict((k,v[:-1]) for k,v in self.candidate[0].items()),self.candidate[1],self.candidate[2][:-1])
        self.assertFalse(self.run_comparison()['gates']['full_duration']['passed'])
    def test_energy_and_current_units(self):
        result=comparison.analyze(*self.base)
        joint=result['windows']['ALL']['joints'][2]
        self.assertAlmostEqual(joint['rms_nm'],10.)
        self.assertAlmostEqual(joint['ideal_current_rms_a'],10/1.44)
        self.assertAlmostEqual(result['windows']['ALL']['absolute_mechanical_work_j'],1792.)
        self.assertAlmostEqual(result['windows']['ALL']['ideal_current_squared_integral_a2_s'],28*4*((2/1.23)**2+(4/1.44)**2+(10/1.44)**2))
    def test_joint_order_reordering_preserves_result(self):
        a,m,s=copy.deepcopy(self.base);m['joint_order']=m['joint_order'][::-1]
        a['estimated_torque']=a['estimated_torque'][:,::-1];a['joint_velocity']=a['joint_velocity'][:,::-1]
        # Status commands have a separately fixed canonical order.
        self.assertEqual(comparison.analyze(a,m,s)['windows'],comparison.analyze(*self.base)['windows'])
    def test_loading_hashes_all_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);a,m,s=self.base
            np.savez(path/'result.npz',**a)
            (path/'result.json').write_text(json.dumps(m));(path/'result.status.json').write_text(json.dumps(s))
            result=comparison.load_run(path)
            self.assertEqual(set(result['source_sha256']),{'result.json','result.status.json','result.npz'})
            self.assertTrue(all(len(v)==64 for v in result['source_sha256'].values()))

if __name__=='__main__':unittest.main()
