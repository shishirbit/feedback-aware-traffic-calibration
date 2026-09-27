import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from ps1.data.provenance import audit_hdf


class ProvenanceTests(unittest.TestCase):
    def test_hdf_audit_records_gaps_zeros_and_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"sample.h5"
            index=pd.to_datetime(["2020-01-01 00:00","2020-01-01 00:05","2020-01-01 00:15"])
            pd.DataFrame([[0.,1.],[2.,np.nan],[3.,4.]],index=index,columns=["a","b"]).to_hdf(path,key="data")
            audit=audit_hdf(path,2,"source")
        self.assertEqual(audit["timestamp_gap_count"],1)
        self.assertEqual(audit["zero_count"],1)
        self.assertEqual(audit["nan_count"],1)
        self.assertEqual(audit["split_bounds"],[0,1,2,2,3])


if __name__=="__main__": unittest.main()
