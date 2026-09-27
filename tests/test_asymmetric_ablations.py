import unittest
import numpy as np
from ps1.evaluation.farcal_online import _method_distribution, _pool


class AsymmetricAblations(unittest.TestCase):
    def setUp(self):
        self.records = {"score": np.array([-4., -1., 2., 8.]), "target": np.array([0, 10, 20, 30]),
            "arrival": np.array([95, 95, 95, 95]), "sensor": np.array([0, 1, 0, 1]),
            "context": np.array([[0.], [2.], [1.], [3.]])}
        self.frozen = {k: v.copy() for k, v in self.records.items()}
        self.parameters = dict(tau=20, bandwidth=1, support=2, stale_decay=20, buffer_bins=2016,
            inflation_cap=3, beta_gap=.1, beta_missing=.1, records_per_sensor_cap=100)

    def distribution(self, ablation, scalable=True, records=None):
        return _method_distribution("far_cal_asymmetric", self.records if records is None else records,
            self.frozen, 100, 0, np.array([0.]), .5, [(0, 1), (0, 1)],
            {**self.parameters, "scalable_local_only": scalable, "ablation": ablation})

    def test_each_component_and_both_pooling_paths(self):
        for scalable in (False, True):
            full = self.distribution("full", scalable)
            no_inflation = self.distribution("no_inflation", scalable)
            np.testing.assert_array_equal(full[0], no_inflation[0])
            np.testing.assert_array_equal(full[1], no_inflation[1])
            self.assertEqual(no_inflation[2], 1.)
            self.assertGreater(full[2], 1.)
            for ablation in ("no_context", "no_age", "no_stale_blend", "arrival_freshness"):
                changed = self.distribution(ablation, scalable)
                np.testing.assert_array_equal(full[0], changed[0])
                self.assertFalse(np.allclose(full[1], changed[1]), ablation)
            spatial = self.distribution("no_spatial", scalable)
            self.assertLess(len(spatial[0]), len(full[0]))
            frozen = self.distribution("frozen_only", scalable)
            changed_records = {**self.records, "score": self.records["score"]*100}
            frozen_changed = self.distribution("frozen_only", scalable, changed_records)
            for x, y in zip(frozen, frozen_changed): np.testing.assert_array_equal(x, y)
            self.assertEqual(frozen[2], 1.)

    def test_empty_live_retains_frozen_fallback(self):
        empty = {k: v[:0] for k, v in self.records.items()}
        for scalable in (False, True):
            scores, weights, _ = self.distribution("no_stale_blend", scalable, empty)
            self.assertTrue(len(scores))
            self.assertAlmostEqual(weights.sum(), 1.)

    def test_arrival_age_keeps_target_block_ess(self):
        records = {**self.records, "target": np.array([0, 0, 20, 20])}
        _, weights, ess = _pool(records, np.ones(4, bool), 100, np.array([0.]), 20, 1,
            use_context=False, freshness="arrival")
        np.testing.assert_allclose(weights, .25)
        self.assertAlmostEqual(ess, 2.)

if __name__ == "__main__": unittest.main()
