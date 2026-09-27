import unittest
import numpy as np
from ps1.evaluation.recovery import recovery_summary
from ps1.evaluation.bootstrap import moving_block_mean_ci,episode_mean_ci


class EvaluationTests(unittest.TestCase):
    def test_recovery_requires_minimum_pairs_and_sustained_windows(self):
        origins=np.arange(20); shape=(20,10)
        valid=np.ones(shape,bool); covered=np.ones(shape,bool)
        score=np.ones(shape); oracle=np.ones(shape)
        result=recovery_summary(origins,covered,score,oracle,valid,0,.1,window_origins=4,min_pairs=40,consecutive=3)
        self.assertEqual(result["coverage_recovery_bins"],3)
        self.assertEqual(result["joint_recovery_bins"],3)
        self.assertEqual(result["minimum_measurable_delay_bins"],3)

    def test_recovery_is_censored_when_coverage_stays_low(self):
        result=recovery_summary(np.arange(10),np.zeros((10,20),bool),np.ones((10,20)),np.ones((10,20)),np.ones((10,20),bool),0,.1,window_origins=4,min_pairs=40)
        self.assertTrue(result["coverage_censored"]); self.assertIsNone(result["coverage_recovery_bins"])

    def test_bootstraps_are_paired_and_seeded(self):
        time=np.arange(20,dtype=float)
        first=moving_block_mean_ci(time,block_bins=4,resamples=50,seed=7)
        second=moving_block_mean_ci(time,block_bins=4,resamples=50,seed=7)
        self.assertEqual(first,second); self.assertEqual(first["estimate"],9.5)
        episodes=episode_mean_ci([1.,2.,3.],resamples=50,seed=7)
        self.assertEqual(episodes["estimate"],2.)


if __name__=="__main__": unittest.main()
