"""Only independent spatial-coordinate algebra; no simulator/model execution."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def comparison_module():
    path = ROOT / "config/experiments/v16_20260929/v16_compare_physx.py"
    if not path.exists():
        raise AssertionError("PhysX coordinate comparison implementation is missing")
    spec = importlib.util.spec_from_file_location("v16_compare_physx", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SpatialComparisonAlgebra(unittest.TestCase):
    def test_actor_local_to_com_world_twist_has_correct_offset_sign(self):
        module = comparison_module()
        R = np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        T = module.base_velocity_transform(R,[0,.5,0],joint_count=1)
        velocity = np.array([1.,0,0,0,0,2,.7])
        np.testing.assert_allclose(T@velocity,[-1,1,0,0,0,2,.7],atol=1e-14)

    def test_moving_frame_transport_keeps_linear_and_centripetal_terms(self):
        module = comparison_module()
        R = np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        transport=module.transport_acceleration(R,[1,0,0,0,0,2],[.5,0,0],joint_count=1)
        np.testing.assert_allclose(transport,[-2,-2,0,0,0,0,0],atol=1e-14)

    def test_mass_force_dual_transform_preserves_energy_and_bias(self):
        module = comparison_module()
        T=module.base_velocity_transform(np.eye(3),[1,0,0],joint_count=0)
        M=np.diag([3.,3,3,1,2,3])
        h=np.array([1.,2,3,4,5,6])
        transport=np.array([.1,.2,.3,0,0,0])
        normalized_M,normalized_h=module.normalize_dynamics(M,h,T,transport)
        expected=np.array([[3,0,0,0,0,0],[0,3,0,0,0,3],[0,0,3,0,-3,0],
                           [0,0,0,1,0,0],[0,0,-3,0,5,0],[0,3,0,0,0,6]],float)
        np.testing.assert_allclose(normalized_M,expected,atol=1e-14)
        np.testing.assert_allclose(normalized_h,[1.3,2.6,3.9,4,1.1,8.6],atol=1e-14)

    def test_fixed_members_aggregate_about_combined_com_once(self):
        module = comparison_module()
        mass,com,inertia=module.aggregate_mass_properties([1,3],[[0,0,0],[2,0,0]],
                                                        [np.diag([1.,2,3]),np.diag([1.,2,3])])
        self.assertEqual(mass,4)
        np.testing.assert_allclose(com,[1.5,0,0],atol=1e-14)
        np.testing.assert_allclose(inertia,np.diag([2,7,9]),atol=1e-14)

    def test_inertia_column_major_is_not_transposed_silently(self):
        module = comparison_module()
        # General asymmetric fixture identifies packing; real inertia is symmetric.
        result=module.unpack_inertias([[1,4,7,2,5,8,3,6,9]])
        np.testing.assert_array_equal(result[0],[[1,2,3],[4,5,6],[7,8,9]])

    def test_same_world_point_survives_shifted_rotated_actor_frames(self):
        module=comparison_module()
        self.assertTrue(callable(getattr(module,"point_in_source_frame",None)),"World-point alignment is missing")
        R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        np.testing.assert_allclose(module.point_in_source_frame([10,20,30],R,[11,22,33]),[2,-1,3],atol=1e-14)
        # Source actor at origin; translated PX actor has local COM -delta.
        actual_com=np.array([1.,0,0])+np.array([-1.,0,0])
        np.testing.assert_allclose(module.point_in_source_frame([0,0,0],np.eye(3),actual_com),[0,0,0],atol=1e-14)


if __name__=="__main__":
    unittest.main(verbosity=2)
