import unittest
import numpy as np
from ps1.data.graphs import local_groups
from ps1.data.prepare import causal_features,episode_windows


class DataPreparationTests(unittest.TestCase):
    def test_groups_are_deterministic_by_hop_then_id(self):
        adjacency=np.zeros((6,6))
        for a,b in ((0,1),(0,2),(1,3),(2,4),(4,5)): adjacency[a,b]=adjacency[b,a]=1
        self.assertEqual(local_groups(adjacency,4)[0],(0,1,2,3,4))

    def test_natural_missingness_is_causally_filled(self):
        speed=np.array([[np.nan],[10.],[np.nan],[20.]])
        valid=np.isfinite(speed)
        filled,age,_=causal_features(speed,valid,[5.])
        np.testing.assert_array_equal(filled[:,0],[5.,10.,10.,20.])
        np.testing.assert_array_equal(age[:,0],[1.,0.,1.,0.])

    def test_windows_never_cross_episode_end(self):
        speed=np.arange(30.,dtype=float)[:,None]; valid=np.ones_like(speed,bool)
        x,y,m=episode_windows(speed,valid,[0.],10.,2.,history=12,horizon=12)
        self.assertEqual(len(x),7)
        self.assertEqual(y[-1,-1,0]*2+10,29.)


if __name__=="__main__": unittest.main()
