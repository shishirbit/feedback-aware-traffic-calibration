import tempfile
import unittest
from pathlib import Path

import numpy as np

from ps1.prediction_cache import _write_chunk,faulted_history,compare_prediction_caches
from ps1.training import TrafficWindowDataset
from ps1.utils.artifacts import sha256_file


class PredictionCacheTests(unittest.TestCase):
    def test_chunk_schema_preserves_invalid_truth_as_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"chunk.npz"; shape=(2,3,4)
            valid=np.ones(shape,dtype=bool); valid[0,0,0]=False
            metadata=_write_chunk(path,[10,11],(np.zeros(shape),np.ones(shape),np.full(shape,2.)),np.full(shape,7.),valid,np.zeros((2,4,2)))
            with np.load(path) as data:
                self.assertTrue(np.isnan(data["truth_mph"][0,0,0]))
                self.assertEqual(data["q50_mph"].dtype,np.float32)
                self.assertEqual(data["issue_bin"].tolist(),[10,11])
            self.assertEqual(metadata["sha256"],sha256_file(path))

    def test_faulted_history_never_uses_unreleased_value(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"data.npz"; rows=200; values=np.arange(rows,dtype=np.float32)[:,None]+10
            valid=np.ones_like(values,dtype=bool); stamps=(np.datetime64("2024-01-01")+np.arange(rows)*np.timedelta64(5,"m")).astype("datetime64[ns]").astype("int64")
            np.savez(path,values_mph=values,original_valid=valid,timestamp_ns=stamps,adjacency=np.ones((1,1)),
                     train_mean=np.float32(10),train_std=np.float32(2),train_sensor_median=np.array([10.],dtype=np.float32))
            dataset=TrafficWindowDataset(path,"test"); test_start=160; arrivals=np.arange(40)[:,None]
            arrivals[0,0]=3
            before,mask_before,_=faulted_history(dataset,arrivals,test_start,160)
            after,mask_after,_=faulted_history(dataset,arrivals,test_start,163)
            self.assertFalse(mask_before[-1,0]); self.assertNotEqual(before[-1,0,0],(values[160,0]-10)/2)
            self.assertTrue(mask_after[-4,0]); self.assertEqual(after[-4,0,0],(values[160,0]-10)/2)

    def test_cache_comparison_requires_exact_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); shape=(1,1,1)
            for name,offset in (("left",0.),("right",1e-4)):
                folder=root/name; folder.mkdir(); path=folder/"chunk.npz"
                np.savez_compressed(path,issue_bin=np.array([1]),q05_mph=np.zeros(shape),q50_mph=np.ones(shape)+offset,q95_mph=np.ones(shape)*2,
                                    truth_mph=np.ones(shape),original_valid=np.ones(shape,dtype=bool),context=np.zeros((1,1,2)))
                (folder/"manifest.json").write_text(__import__('json').dumps({"chunks":[{"file":"chunk.npz","sha256":sha256_file(path)}]}),encoding="utf-8")
            result=compare_prediction_caches(root/"left",root/"right",root/"result.json")
            self.assertFalse(result["equivalent"]); self.assertFalse(result["comparisons"]["q50_mph"]["exact"])


if __name__=="__main__": unittest.main()
