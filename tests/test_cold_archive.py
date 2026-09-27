import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ps1.evaluation.farcal_online import _continuous_calibration_records
from ps1.evaluation.online_compare import _initialize_store
from ps1.utils.artifacts import sha256_file


class ColdArchiveTests(unittest.TestCase):
    def test_lost_calibration_label_is_absent_from_both_calibrators(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calibration = root / "calibration"
            fault = root / "fault"
            calibration.mkdir()
            fault.mkdir()
            q50 = np.zeros((2, 1, 2), dtype=np.float32)
            q05 = np.full_like(q50, -1)
            q95 = np.full_like(q50, 1)
            truth = np.ones_like(q50)
            valid = np.ones((2, 1, 2), dtype=bool)
            issues = np.array([10, 11], dtype=np.int32)
            chunk = calibration / "chunk-0000.npz"
            np.savez_compressed(chunk, q05_mph=q05, q50_mph=q50, q95_mph=q95,
                                truth_mph=truth, original_valid=valid, issue_bin=issues,
                                context=np.zeros((2, 2, 2), dtype=np.float32))
            (calibration / "manifest.json").write_text(json.dumps({"chunks": [
                {"file": chunk.name, "sha256": sha256_file(chunk)}]}), encoding="utf-8")
            (fault / "fault_manifest.json").write_text(json.dumps({"partition_start_row": 11}), encoding="utf-8")
            np.savez_compressed(fault / "release_schedule.npz",
                                feedback_arrival=np.array([[-1, 0], [1, 1]]))
            with patch("ps1.evaluation.farcal_online._cache_arrays", return_value={
                "q05_mph": q05, "q50_mph": q50, "q95_mph": q95,
                "truth_mph": truth, "original_valid": valid, "issue_bin": issues,
                "context": np.zeros((2, 2, 2), dtype=np.float32)}):
                records = _continuous_calibration_records(calibration, {}, (1,), fault, 15)
            self.assertEqual(len(records[1]["score"]), 3)
            self.assertFalse(((records[1]["target"] == 11) & (records[1]["sensor"] == 0)).any())
            store = _initialize_store(calibration, (1,), 2, 15, 20, fault)
            self.assertEqual(sum(len(node) for node in store.sorted[1]), 3)


if __name__ == "__main__":
    unittest.main()
