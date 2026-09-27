import unittest
from ps1.calibration.far_cal import Residual
from ps1.calibration.received import ReceivedResidualCalibrator
from ps1.online.events import ReleaseEvent
from ps1.online.ledger import Forecast


def records(count=50):
    return [Residual(f"a{i}",0,1,-count+i,-count+i,float(i+1),(0.,0.)) for i in range(count)]


class ReceivedCalibratorTests(unittest.TestCase):
    def test_fixed_small_sample_is_explicitly_unbounded(self):
        calibrator=ReceivedResidualCalibrator([[0]],"frozen"); calibrator.initialize(records(4))
        intervals,_=calibrator.predict_one(0,0,1,20.,1.,(0.,0.),0.)
        self.assertEqual(intervals[0][2],float("inf")); self.assertEqual(intervals[1][2],float("inf"))

    def test_delayed_feedback_enters_once_after_release(self):
        calibrator=ReceivedResidualCalibrator([[0]],"rolling"); calibrator.initialize(records())
        forecast=Forecast("new","episode",0,1,1,0,10.,1.,((.9,0.,20.),(.95,0.,20.)),(0.,0.))
        event=ReleaseEvent("event","feedback","episode",1,5,0,12.)
        with self.assertRaises(ValueError): calibrator.observe(forecast,event,4)
        calibrator.observe(forecast,event,5); calibrator.observe(forecast,event,5)
        self.assertEqual(sum(record.forecast_id=="new" for record in calibrator.live),1)

    def test_expiration_uses_target_not_arrival(self):
        calibrator=ReceivedResidualCalibrator([[0]],"exponential",buffer_bins=10); calibrator.initialize(records())
        forecast=Forecast("old","episode",0,1,1,0,10.,1.,((.9,0.,20.),(.95,0.,20.)),(0.,0.))
        event=ReleaseEvent("event","feedback","episode",1,100,0,12.)
        calibrator.observe(forecast,event,100); calibrator.expire_by_target_time(100)
        self.assertFalse(any(record.forecast_id=="old" for record in calibrator.live))

    def test_state_roundtrip(self):
        calibrator=ReceivedResidualCalibrator([[0]],"rolling"); calibrator.initialize(records())
        clone=ReceivedResidualCalibrator([[0]]); clone.load_state_dict(calibrator.state_dict())
        self.assertEqual(calibrator.predict_one(0,0,1,20.,1.,(0.,0.),0.),clone.predict_one(0,0,1,20.,1.,(0.,0.),0.))


if __name__=="__main__": unittest.main()
