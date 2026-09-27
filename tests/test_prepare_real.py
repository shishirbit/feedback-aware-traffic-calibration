import pickle
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from ps1.data.prepare_real import prepare_hdf


class RealPreparationTests(unittest.TestCase):
    def test_gap_and_zero_become_invalid_without_changing_raw_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); hdf=root/"data.h5"; graph=root/"graph.pkl"; output=root/"prepared.npz"
            index=pd.to_datetime(["2020-01-01 00:00","2020-01-01 00:05","2020-01-01 00:15","2020-01-01 00:20","2020-01-01 00:25"])
            pd.DataFrame([[1.,2.],[3.,4.],[5.,6.],[7.,8.],[9.,0.]],index=index,columns=["a","b"]).to_hdf(hdf,key="data")
            with graph.open("wb") as handle: pickle.dump((["a","b"],{"a":0,"b":1},np.eye(2)),handle)
            manifest=prepare_hdf(hdf,graph,output,2)
            self.assertEqual(manifest["inserted_missing_rows"],1)
            self.assertEqual(manifest["boundary_timestamps"][1],"2020-01-01T00:20:00")
            with np.load(output) as arrays:
                self.assertFalse(arrays["original_valid"][2].any())
                self.assertFalse(arrays["original_valid"][-1,1])


if __name__=="__main__": unittest.main()
