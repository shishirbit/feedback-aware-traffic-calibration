import unittest

from ps1.calibration.aci_adapter import ArrivalBatchACI
from ps1.calibration.far_cal import Residual
from ps1.online.events import ReleaseEvent
from ps1.online.ledger import Forecast


def archive():
    return [Residual(f"a{i}", 0, 1, -60 + i, -60 + i, 1. + i / 100., (0., 0.)) for i in range(60)]


def forecast(name, sensor, lower=8., upper=12.):
    return Forecast(name, "e", 0, 1, 1, sensor, 10., 1.,
                    ((.90, lower, upper), (.95, lower, upper)), (0., 0.))


def event(name, sensor, value, arrival=4):
    return ReleaseEvent(name, "feedback", "e", 1, arrival, sensor, value)


class ArrivalBatchACITests(unittest.TestCase):
    def test_no_update_before_release(self):
        cal = ArrivalBatchACI([[0]]); cal.initialize(archive())
        with self.assertRaises(ValueError):
            cal.observe(forecast("f", 0), event("x", 0, 20.), 3)
        self.assertAlmostEqual(cal.control[(1, .90)], .10)
        self.assertFalse(cal.pending)

    def test_batch_update_uses_mean_miss_and_is_order_invariant(self):
        states = []
        pairs = [(forecast("hit", 0), event("eh", 0, 10.)),
                 (forecast("miss", 1), event("em", 1, 20.))]
        for ordered in (pairs, reversed(pairs)):
            cal = ArrivalBatchACI([[0], [1]], learning_rates={.90: .01, .95: .02})
            cal.initialize(archive())
            for issued, released in ordered:
                cal.observe(issued, released, 4)
            cal.finish_arrival_batch(4)
            states.append((dict(cal.control), dict(cal.update_count)))
        self.assertEqual(states[0], states[1])
        self.assertAlmostEqual(states[0][0][(1, .90)], .10 + .01 * (.10 - .5))
        self.assertAlmostEqual(states[0][0][(1, .95)], .05 + .02 * (.05 - .5))
        self.assertEqual(states[0][1][(1, .90)], 1)

    def test_miss_uses_original_issued_bounds(self):
        cal = ArrivalBatchACI([[0]]); cal.initialize(archive())
        cal.observe(forecast("f", 0, 0., 9.), event("x", 0, 10.), 4)
        cal.finish_arrival_batch(4)
        self.assertLess(cal.control[(1, .90)], .10)

    def test_state_roundtrip(self):
        cal = ArrivalBatchACI([[0]]); cal.initialize(archive())
        intervals, _ = cal.predict_one(0, 0, 1, 10., 1., (0., 0.), 0.)
        clone = ArrivalBatchACI([[0]]); clone.load_state_dict(cal.state_dict())
        self.assertEqual(intervals, clone.predict_one(0, 0, 1, 10., 1., (0., 0.), 0.)[0])


if __name__ == "__main__":
    unittest.main()
