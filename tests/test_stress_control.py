import unittest

import numpy as np

from ps1.online.faults import generate_fault_schedule
from ps1.online.stress import (loss_variant_from_source, matched_independent_control,
                               mnar_schedule, one_factor_schedule)


class MatchedControlTests(unittest.TestCase):
    def test_preserves_per_bin_severity_and_breaks_fixed_group(self):
        adjacency = np.zeros((30, 30))
        for index in range(29):
            adjacency[index, index + 1] = adjacency[index + 1, index] = 1
        for scenario in ("C3", "C4"):
            source = generate_fault_schedule(scenario, 288, adjacency, 101,
                                             connected_fraction=.2, loss_within=.5)
            control = matched_independent_control(source, 999)
            np.testing.assert_array_equal(control.input_arrival, control.feedback_arrival)
            for event in source.events:
                for bin_index in range(event["start_bin"], event["end_bin"]):
                    original = source.input_arrival[bin_index]
                    independent = control.input_arrival[bin_index]
                    self.assertEqual(int((original != bin_index).sum()),
                                     int((independent != bin_index).sum()))
                    np.testing.assert_array_equal(np.sort(original[original != bin_index]),
                                                  np.sort(independent[independent != bin_index]))
            chosen = control.events[0]["affected_sensors_by_bin"]
            self.assertNotEqual(chosen[0], chosen[1])

    def test_one_factor_and_loss_stress_keep_coupled_channels(self):
        adjacency = np.zeros((20, 20))
        for index in range(19):
            adjacency[index, index + 1] = adjacency[index + 1, index] = 1
        source = generate_fault_schedule("C4", 288, adjacency, 101)
        longer = one_factor_schedule("C4", 288, adjacency, 101, "duration_bins", 12)
        self.assertEqual(longer.events[0]["end_bin"] - longer.events[0]["start_bin"], 12)
        lossless = loss_variant_from_source(source, 101, 0.)
        np.testing.assert_array_equal(lossless.events[0]["affected_sensors"], source.events[0]["affected_sensors"])
        self.assertFalse((lossless.input_arrival < 0).any())
        np.testing.assert_array_equal(lossless.input_arrival, lossless.feedback_arrival)

    def test_mnar_adds_loss_only_at_true_low_speed(self):
        adjacency = np.zeros((20, 20))
        for index in range(19):
            adjacency[index, index + 1] = adjacency[index + 1, index] = 1
        source = generate_fault_schedule("C3", 288, adjacency, 101)
        values = np.full((288, 20), 50., dtype=float)
        event = source.events[0]
        values[event["start_bin"]:event["end_bin"], event["affected_sensors"]] = 10.
        stressed = mnar_schedule(source, values, np.full(20, 20.), 0, 202, 1.)
        self.assertEqual(stressed.events[0]["additional_mnar_lost_packets"],
                         6 * len(event["affected_sensors"]))
        self.assertEqual(stressed.events[1]["additional_mnar_lost_packets"], 0)


if __name__ == "__main__":
    unittest.main()
