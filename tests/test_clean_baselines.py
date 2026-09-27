import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ps1.evaluation.clean_baselines import evaluate_clean_baselines,evaluate_simple_controls
from ps1.utils.artifacts import sha256_file


def cache(directory,partition,truth,low,point,high,valid):
    directory.mkdir(); chunk=directory/"chunk-0000.npz"
    np.savez_compressed(chunk,issue_bin=np.arange(len(truth)),q05_mph=low,q50_mph=point,q95_mph=high,
                        truth_mph=np.where(valid,truth,np.nan),original_valid=valid,context=np.zeros((len(truth),truth.shape[2],2)))
    manifest={"schema_version":1,"status":"completed","partition":partition,"origins":len(truth),"horizons":truth.shape[1],
              "sensors":truth.shape[2],"checkpoint_sha256":"same","checkpoint_training_seed":11,
              "chunks":[{"file":chunk.name,"sha256":sha256_file(chunk)}]}
    (directory/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")


class CleanBaselineTests(unittest.TestCase):
    def test_exact_streaming_metrics_and_invalid_exclusion(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); shape=(20,3,1); truth=np.full(shape,10.); point=np.full(shape,9.)
            low=np.full(shape,8.); high=np.full(shape,10.); valid=np.ones(shape,dtype=bool); valid[0,0,0]=False
            cache(root/"cal","calibration",truth,low,point,high,valid)
            cache(root/"test","test",truth,low,point,high,valid)
            result=evaluate_clean_baselines(root/"cal",root/"test",root/"result.json",(1,2,3))
            self.assertEqual(result["point"]["1"]["count"],19)
            self.assertAlmostEqual(result["point"]["2"]["mae_mph"],1.)
            self.assertAlmostEqual(result["intervals"]["raw_q05_q95"]["3"]["picp"],1.)
            self.assertTrue((root/"result.json").exists())

    def test_simple_controls_use_test_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); path=root/"demo.npz"; rows=200; sensors=2
            values=np.full((rows,sensors),20.,dtype=np.float32); valid=np.ones_like(values,dtype=bool)
            stamps=(np.datetime64("2024-01-01")+np.arange(rows)*np.timedelta64(5,"m")).astype("datetime64[ns]").astype("int64")
            np.savez(path,values_mph=values,original_valid=valid,timestamp_ns=stamps,train_sensor_median=np.full(sensors,20.))
            result=evaluate_simple_controls(path,root/"controls.json",(3,6,12))
            self.assertEqual(result["methods"]["last_available"]["3"]["mae_mph"],0.)
            self.assertEqual(result["methods"]["historical_time_of_week"]["12"]["rmse_mph"],0.)


if __name__=="__main__": unittest.main()
