"""Repeated-candidate selection algebra only; no simulator imports."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np


def robust_module():
    path=Path(__file__).resolve().parents[1]/"v16_cem_robust.py"
    if not path.exists():
        raise AssertionError("Required repeated-candidate CEM helper is missing")
    spec=importlib.util.spec_from_file_location("v16_cem_robust",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RepeatedQualification(unittest.TestCase):
    def test_all_four_replicas_must_pass_alive_progress_and_full_cycle(self):
        scores=np.array([[30,20,10,0],[200,190,180,170],[40,50,60,70],[80,90,100,110]],float)
        alive=np.ones((4,4),bool);alive[1,3]=False
        progress=np.full((4,4),.6);progress[2,2]=.49
        steps=np.full((4,4),4);steps[3,0]=3
        aggregate,qualified,diagnostics=robust_module().aggregate(
            scores.ravel(),alive.ravel(),progress.ravel(),steps.ravel(),2.,4)
        np.testing.assert_array_equal(aggregate,[0.,170.,40.,80.])
        np.testing.assert_array_equal(qualified,[True,False,False,False])
        np.testing.assert_array_equal(diagnostics["alive_count"],[4,3,4,4])
        np.testing.assert_allclose(diagnostics["progress_min"],[.6,.6,.49,.6])
        np.testing.assert_array_equal(diagnostics["completed_steps_min"],[4,4,4,3])
        self.assertEqual(diagnostics["qualified_count"],1)
        self.assertEqual(diagnostics["unique_count"],4)
        self.assertEqual(diagnostics["replicas_per_candidate"],4)

    def test_repeat_grouping_matches_np_repeat_and_minimum_is_not_mean_or_quantile(self):
        unique=np.array([[1.,2.],[3.,4.]])
        population=np.repeat(unique,4,axis=0)
        np.testing.assert_array_equal(population[:4],np.repeat(unique[:1],4,axis=0))
        scores=np.array([30,20,10,0, 4,100,50,20],float)
        aggregate,qualified,diagnostics=robust_module().aggregate(
            scores,np.ones(8,bool),np.ones(8),np.full(8,4),2.,4)
        np.testing.assert_array_equal(aggregate,[0.,4.])
        np.testing.assert_allclose(diagnostics["score_q25"],[7.5,16.])
        np.testing.assert_array_equal(qualified,[True,True])

    def test_progress_boundary_and_four_steps_are_inclusive(self):
        _,qualified,_=robust_module().aggregate([1,2,3,4],[1,1,1,1],
                                              [.5]*4,[4.]*4,2.,4)
        self.assertTrue(qualified[0])

    def test_dead_replica_cannot_be_saved_by_large_positive_score_or_progress(self):
        aggregate,qualified,_=robust_module().aggregate([1e6]*4,[1,1,0,1],
            [100.]*4,[100]*4,2.,4)
        self.assertEqual(aggregate[0],1e6)
        self.assertFalse(qualified[0])

    def test_aggregation_does_not_mutate_inputs(self):
        arrays=[np.arange(8,dtype=float),np.ones(8,bool),np.ones(8),np.full(8,4)]
        copies=[value.copy() for value in arrays]
        robust_module().aggregate(*arrays,2.,4)
        for value,original in zip(arrays,copies):
            np.testing.assert_array_equal(value,original)

    def test_invalid_replica_arrays_counters_targets_and_repeats_fail_fast(self):
        base=dict(scores=np.ones(8),alive=np.ones(8,bool),progress=np.ones(8),
                  completed_steps=np.full(8,4),target_distance=2.,repeats=4)
        for update in ({"scores":np.ones(7)},{"alive":np.ones((8,1))},
                       {"progress":np.ones(4)},{"alive":[1,1,1,2]*2},
                       {"scores":np.full(8,np.nan)},{"progress":np.full(8,np.inf)},
                       {"completed_steps":np.full(8,3.5)},{"completed_steps":np.full(8,-1)},
                       {"target_distance":-1.},{"target_distance":np.inf},
                       {"repeats":0},{"repeats":True},{"repeats":1.5},{"scores":[]}):
            values=base.copy();values.update(update)
            with self.subTest(update=list(update)),self.assertRaises(ValueError):
                robust_module().aggregate(**values)


class QualifiedCEMUpdate(unittest.TestCase):
    @staticmethod
    def fixture():
        samples=np.zeros((10,2))
        samples[[0,3,6]]=[[2,4],[6,12],[8,16]]
        scores=np.arange(10,dtype=float);scores[[0,3,6,9]]=[10,20,30,1e6]
        qualified=np.zeros(10,bool);qualified[[0,3,6]]=True
        return dict(samples=samples,aggregate_scores=scores,qualified=qualified,
                    lower=np.array([0.,0.]),upper=np.array([10.,20.]),
                    oldmean=np.array([5.,10.]),oldstd=np.array([1.,2.]))

    def test_only_qualified_elites_move_distribution_even_when_failure_score_is_larger(self):
        mean,std,details=robust_module().update(**self.fixture())
        np.testing.assert_allclose(mean,[6.4,12.8])
        np.testing.assert_allclose(std,[1.,2.])
        np.testing.assert_array_equal(details["elite_indices"],[6,3])
        self.assertEqual(details["qualified_count"],3)
        self.assertEqual(details["elite_count"],2)
        self.assertTrue(details["updated_from_qualified"])

    def test_elite_count_uses_unique_population_ceiling_and_available_qualifiers(self):
        for unique_n,qualified_n,expected in ((13,2,2),(21,6,5)):
            values=dict(samples=np.arange(unique_n)[:,None].astype(float),
                aggregate_scores=np.arange(unique_n,dtype=float),
                qualified=np.arange(unique_n)<qualified_n,lower=np.array([0.]),
                upper=np.array([float(unique_n)]),oldmean=np.array([5.]),oldstd=np.array([1.]))
            _,_,details=robust_module().update(**values)
            self.assertEqual(details["elite_count"],expected)
            self.assertEqual(details["unique_count"],unique_n)

    def test_standard_deviation_floor_is_five_percent_of_each_parameter_range(self):
        values=self.fixture();values["samples"][:]=[5.,10.];values["oldstd"][:]=.01
        _,std,_=robust_module().update(**values)
        np.testing.assert_allclose(std,[.5,1.])

    def test_zero_or_one_qualifier_keeps_mean_and_widens_without_using_lucky_sample(self):
        for count in (0,1):
            values=self.fixture();values["qualified"][:]=False
            values["qualified"][:count]=True
            mean,std,details=robust_module().update(**values)
            np.testing.assert_array_equal(mean,values["oldmean"])
            np.testing.assert_allclose(std,[1.25,2.5])
            self.assertEqual(details["qualified_count"],count)
            self.assertEqual(details["elite_count"],0)
            self.assertEqual(details["elite_indices"].size,0)
            self.assertFalse(details["updated_from_qualified"])

    def test_repeated_failure_does_not_shrink_or_move_mean_at_expansion_ceiling(self):
        values=self.fixture();values["qualified"][:]=False
        means=[];stds=[]
        for _ in range(8):
            mean,std,_=robust_module().update(**values)
            means.append(mean.copy());stds.append(std.copy())
            values["oldmean"],values["oldstd"]=mean,std
        np.testing.assert_array_equal(means,np.repeat([[5.,10.]],8,axis=0))
        self.assertTrue((np.diff(stds,axis=0)>=0).all())
        np.testing.assert_array_equal(stds[-1],[2.,4.])

    def test_already_wide_standard_deviation_is_retained_when_qualification_fails(self):
        values=self.fixture();values["qualified"][:]=False
        values["oldstd"][:]=[3.,5.]
        _,std,_=robust_module().update(**values)
        np.testing.assert_array_equal(std,[3.,5.])

    def test_qualified_update_is_not_capped_at_failure_expansion_ceiling(self):
        mean,std,_=robust_module().update(np.array([[0.],[10.]]),np.array([1.,2.]),
            np.array([True,True]),np.array([0.]),np.array([10.]),np.array([5.]),np.array([1.]))
        np.testing.assert_allclose(mean,[5.])
        np.testing.assert_allclose(std,[3.8])

    def test_convex_mean_at_parameter_boundary_remains_valid_on_next_update(self):
        # .3*.04+.7*.04 rounds below .04, which would fail the next update's
        # strict bound check. These are actual step/duty/frequency bound units.
        lower=np.array([.04,.8,6.,.1]);upper=lower+1.
        samples=np.repeat(lower[None],2,axis=0)
        mean,std,_=robust_module().update(samples,[1.,2.],[True,True],lower,upper,
                                          lower,np.full(4,.1))
        self.assertTrue((mean>=lower).all() and (mean<=upper).all())
        robust_module().update(samples,[1.,2.],[True,True],lower,upper,mean,std)

    def test_tied_scores_have_deterministic_original_index_order_and_inputs_are_unchanged(self):
        values=self.fixture();values["aggregate_scores"][:]=1.
        originals={k:v.copy() for k,v in values.items()}
        _,_,details=robust_module().update(**values)
        np.testing.assert_array_equal(details["elite_indices"],[0,3])
        for key,value in originals.items():
            np.testing.assert_array_equal(values[key],value)

    def test_invalid_shapes_finite_values_bounds_and_standard_deviation_fail_fast(self):
        for update in ({"samples":np.zeros(2)},{"aggregate_scores":np.zeros(4)},
                       {"qualified":np.ones((10,1))},{"qualified":np.full(10,2)},
                       {"aggregate_scores":np.full(10,np.nan)},{"lower":np.ones(3)},
                       {"upper":np.zeros(2)},{"oldmean":np.full(2,30.)},
                       {"oldstd":np.zeros(2)},{"samples":np.full((10,2),-1.)},
                       {"oldstd":np.full(2,np.inf)}):
            values=self.fixture();values.update(update)
            with self.subTest(update=list(update)),self.assertRaises(ValueError):
                robust_module().update(**values)


if __name__=="__main__":
    unittest.main()
