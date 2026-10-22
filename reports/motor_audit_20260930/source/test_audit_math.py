import unittest
import numpy as np
from audit_math import joint_statistics

class AuditMathTests(unittest.TestCase):
    def test_signed_rms_mask_and_duration(self):
        torque = np.array([[[3.], [4.]], [[-3.], [999.]], [[0.], [999.]]])
        valid = np.array([[1,1],[1,0],[1,0]], dtype=bool)
        row = joint_statistics(torque, torque, valid, .005, [2.], [4.], [4.], [1.])[0]
        self.assertAlmostEqual(row['torque_rms_nm'], np.sqrt(34/4))
        self.assertEqual(row['torque_peak_nm'], 4.)
        self.assertEqual(row['rated_exceed_fraction'], .75)
        self.assertEqual(row['sim_limit_exceed_fraction'], 0.)
        self.assertAlmostEqual(row['rated_exceed_max_contiguous_s'], .01)
        self.assertEqual(row['valid_samples'], 4)
        self.assertAlmostEqual(row['current_rms_ideal_a'], np.sqrt(34/4))
    def test_current_conversion_and_requested_limit(self):
        torque=np.array([[[1.23, 1.44]]])
        requested=np.array([[[30., 55.]]])
        rows=joint_statistics(torque, requested, np.ones((1,1),bool), .005,
                              [8.3,18], [24.8,48], [24.8,53], [1.23,1.44])
        for row in rows:
            self.assertAlmostEqual(row['current_peak_ideal_a'], 1.)
            self.assertEqual(row['requested_sim_limit_exceed_fraction'],1.)
    def test_nonfinite_valid_sample_rejected(self):
        a=np.array([[[np.nan]]])
        with self.assertRaises(ValueError):
            joint_statistics(a,a,np.ones((1,1),bool),.005,[1],[2],[2],[1])
    def test_empty_not_zero_load(self):
        a=np.zeros((2,1,1))
        with self.assertRaises(ValueError):
            joint_statistics(a,a,np.zeros((2,1),bool),.005,[1],[2],[2],[1])

if __name__=='__main__': unittest.main()
