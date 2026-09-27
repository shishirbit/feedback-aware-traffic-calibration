import unittest
import numpy as np
from ps1.evaluation.online_compare import SortedNodeResiduals,OnlineMetricSums,_interval_arrays
from ps1.evaluation.bootstrap import moving_block_mean_ci


class OnlineCompareTests(unittest.TestCase):
    def test_sorted_store_expires_by_target_and_matches_higher_quantile(self):
        store=SortedNodeResiduals((3,),1,buffer_bins=5)
        store.insert(3,1,[0],[1.]); store.insert(3,4,[0],[4.]); store.insert(3,5,[0],[2.])
        self.assertEqual(store.quantile(3,.5)[0],2.)
        store.expire(7)
        self.assertEqual(store.sorted[3][0],[2.,4.])

    def test_unbounded_intervals_are_explicit(self):
        sums=OnlineMetricSums(); sums.add(np.array([1.]),np.array([0.]),np.array([np.inf]),np.array([True]),.1)
        result=sums.result(); self.assertIsNone(result["mpiw_mph"]); self.assertEqual(result["unbounded_count"],1)

    def test_interval_array_masks_and_penalty(self):
        truth=np.array([[0.,3.]]); lower=np.array([[1.,1.]]); upper=np.array([[2.,2.]]); valid=np.array([[True,False]])
        score,covered,width=_interval_arrays(truth,lower,upper,valid,.1)
        self.assertEqual(score[0,0],21.); self.assertEqual(covered[0,0],0.); self.assertTrue(np.isnan(width[0,1]))

    def test_block_bootstrap_retains_cells_and_allows_empty_bin(self):
        values=np.array([[1.,3.],[np.nan,np.nan],[5.,np.nan],[7.,9.]])
        result=moving_block_mean_ci(values,block_bins=2,resamples=20,seed=4)
        self.assertEqual(result["estimate"],5.)


if __name__=="__main__": unittest.main()
