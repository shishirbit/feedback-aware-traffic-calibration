import tempfile
import unittest
from pathlib import Path

import numpy as np

from ps1.training import TrafficWindowDataset, _calendar_features, _connected_nodes


class TrainingDataTests(unittest.TestCase):
    def fixture(self, path):
        rows, sensors = 200, 4
        values = np.arange(rows*sensors, dtype=np.float32).reshape(rows, sensors) + 1
        valid = np.ones_like(values, dtype=bool); valid[10, 0] = False; values[10, 0] = 0
        timestamps = np.datetime64("2024-01-01T00:00") + np.arange(rows)*np.timedelta64(5, "m")
        adjacency = np.eye(sensors, k=1, dtype=np.float32) + np.eye(sensors, k=-1, dtype=np.float32)
        np.savez(path, values_mph=values, original_valid=valid, timestamp_ns=timestamps.astype("datetime64[ns]").astype("int64"),
                 adjacency=adjacency, train_mean=np.float32(100), train_std=np.float32(20),
                 train_sensor_median=np.arange(sensors, dtype=np.float32)+10)

    def test_targets_remain_inside_partition_and_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"data.npz"; self.fixture(path)
            dataset = TrafficWindowDataset(path, "tune")
            self.assertEqual(int(dataset.issues[0]), 119)
            self.assertLessEqual(int(dataset.issues[-1])+12, 140)
            x, y, valid = dataset[0]
            self.assertEqual(tuple(x.shape), (12, 4, 7)); self.assertEqual(tuple(y.shape), (12, 4))
            self.assertEqual(y.shape, valid.shape)

    def test_augmentation_is_deterministic_by_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"data.npz"; self.fixture(path)
            left = TrafficWindowDataset(path, "train", augment=True, seed=5)
            right = TrafficWindowDataset(path, "train", augment=True, seed=5)
            self.assertTrue(np.array_equal(left[3][0], right[3][0]))
            left.set_epoch(2); right.set_epoch(2)
            self.assertTrue(np.array_equal(left[3][0], right[3][0]))

    def test_episode_windows_and_causal_state_reset_at_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"episodes.npz"; rows,sensors=60,2
            values=np.arange(rows*sensors,dtype=np.float32).reshape(rows,sensors)+1
            valid=np.ones_like(values,dtype=bool); values[30]=999; valid[30]=False
            timestamps=(np.datetime64("2024-01-01")+np.arange(rows)*np.timedelta64(5,"m")).astype("datetime64[ns]").astype("int64")
            np.savez(path,values_mph=values,original_valid=valid,timestamp_ns=timestamps,
                     adjacency=np.eye(sensors,dtype=np.float32),train_mean=np.float32(1),train_std=np.float32(1),
                     train_sensor_median=np.array([7,8],dtype=np.float32),episode_index=np.repeat([0,1],30),
                     partition_code=np.zeros(rows,dtype=np.uint8),warmup_bins=np.int16(2))
            dataset=TrafficWindowDataset(path,"train",history=12,horizon=12)
            self.assertEqual(dataset.issues.tolist(),[13,14,15,16,17,43,44,45,46,47])
            self.assertEqual(dataset.filled[30].tolist(),[7,8])
            self.assertEqual(dataset.age[30].tolist(),[1,1])

    def test_calendar_uses_actual_weekday(self):
        stamps = np.array([np.datetime64("2024-01-01T00:00"), np.datetime64("2024-01-02T00:00")]).astype("datetime64[ns]").astype("int64")
        features = _calendar_features(stamps)
        self.assertAlmostEqual(float(features[0, 0]), 0.)
        self.assertFalse(np.array_equal(features[0, 2:], features[1, 2:]))

    def test_connected_selection(self):
        adjacency = np.eye(5, k=1) + np.eye(5, k=-1)
        self.assertEqual(set(_connected_nodes(adjacency, 2, 3)), {1, 2, 3})
        disconnected = np.zeros((5, 5))
        self.assertEqual(len(_connected_nodes(disconnected, 2, 4)), 4)


if __name__ == "__main__":
    unittest.main()
