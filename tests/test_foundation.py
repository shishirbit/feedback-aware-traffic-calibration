import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import tempfile
import numpy as np
from ps1.online.events import Observation, ReleaseEvent
from ps1.online.gateway import ReportingGateway
from ps1.online.history import InputHistory
from ps1.online.ledger import Forecast, ForecastLedger
from ps1.calibration.frozen import conformal_quantile, weighted_quantile
from ps1.calibration.far_cal import FARCal, Residual
from ps1.evaluation.farcal_online import _method_widths, _calibration_records
from ps1.utils.artifacts import sha256_file
from ps1.data.windows import origins, training_statistics
from ps1.evaluation.metrics import interval_metrics


class FoundationTests(unittest.TestCase):
    def event(self, channel="input", observed=1, arrival=4, value=30., valid=True):
        return ReleaseEvent("event", channel, "test", observed, arrival, 0, value, valid)

    def forecast(self):
        return Forecast("f", "test", 0, 12, 12, 0, 20., 2., ((.9, 10., 30.), (.95, 5., 35.)), (1., 0.))

    def calibrator(self):
        cal = FARCal([[0]])
        cal.initialize([Residual(f"a{i}", 0, 12, -60+i, -60+i, float(i+1), (1., 0.)) for i in range(50)])
        return cal

    def test_release_horizon_plus_delay(self):
        gateway = ReportingGateway(lambda o, c: o.observed_bin + 3)
        gateway.ingest_truth(Observation("test", 0, 12, 25.))
        self.assertEqual(gateway.release_due(14), [])
        self.assertEqual(len(gateway.release_due(15)), 2)
        self.assertEqual(gateway.release_due(15), [])
        with self.assertRaises(ValueError):
            self.event(observed=10, arrival=9)

    def test_unreleased_truth_invariance(self):
        windows = []
        for value in (30., 1000.):
            gateway = ReportingGateway(lambda o, c: 10)
            gateway.ingest_truth(Observation("test", 0, 1, value))
            history = InputHistory("test", [50.])
            for event in gateway.release_due(5):
                if event.channel == "input":
                    history.ingest(event, 5)
            windows.append(history.causal_window(5))
        for a,b in zip(*windows):
            np.testing.assert_array_equal(a,b)

    def test_history_never_fills_past_from_future_slot(self):
        history = InputHistory("test", [50.])
        history.ingest(self.event(observed=3, arrival=4), 4)
        x, mask, age = history.causal_window(4, 4)
        np.testing.assert_array_equal(x[:,0], [50.,50.,30.,30.])
        np.testing.assert_array_equal(mask[:,0], [False,False,True,False])
        self.assertEqual(age[-1,0], 1)

    def test_forecast_immutable_feedback_deduplicated(self):
        forecast, ledger, cal = self.forecast(), ForecastLedger(), self.calibrator()
        ledger.append(forecast)
        with self.assertRaises(FrozenInstanceError):
            forecast.point = 99
        event = self.event("feedback",12,15,45.)
        ledger.ingest_feedback(event, cal, 15)
        ledger.ingest_feedback(replace(event, event_id="duplicate"), cal, 15)
        self.assertEqual(len(ledger.feedback_matches), 1)
        self.assertEqual(cal.live[-1].score, 12.5)
        self.assertEqual(ledger.feedback_matches[0][-1], ((.9,True),(.95,True)))

    def test_invalid_feedback_excluded(self):
        cal = self.calibrator()
        cal.observe(self.forecast(), self.event("feedback",12,15,0.,False),15)
        self.assertEqual(len(cal.live),50)

    def test_target_time_expiration(self):
        cal = self.calibrator()
        cal.observe(self.forecast(), self.event("feedback",12,3000,45.),3000)
        cal.expire_by_target_time(3000)
        self.assertEqual(cal.live, [])
        self.assertEqual(len(cal.frozen),50)

    def test_split_and_training_only_statistics(self):
        self.assertEqual(list(origins(60,70,history=12,horizon=3)),list(range(59,67)))
        values = np.array([[1.],[3.],[1000.]])
        stats = training_statistics(values,np.ones_like(values,bool),2)
        self.assertEqual(stats["mean"],2.)
        self.assertEqual(stats["medians"][0],2.)

    def test_hand_quantiles(self):
        self.assertEqual(conformal_quantile([1,2,3,4], .4),3.)
        self.assertEqual(conformal_quantile([1,2,3,4], .1),float("inf"))
        self.assertEqual(weighted_quantile([1,3,10],[.45,.45,.10],.9),3.)

    def test_block_support_not_sensor_count(self):
        cal = self.calibrator()
        records=[Residual(str(i),i,12,0,0,1.,(1.,0.)) for i in range(100)]
        _,_,ess=cal._pool(records,1,np.array([1.,0.]))
        self.assertEqual(ess,1.)

    def test_nested_fallback_and_state_roundtrip(self):
        cal = self.calibrator()
        a=cal.predict_one(3000,0,12,20.,2.,(1.,0.),1.)
        self.assertEqual(a[1]["eta"],0.)
        self.assertLessEqual(a[0][1][1],a[0][0][1])
        self.assertGreaterEqual(a[0][1][2],a[0][0][2])
        restored=FARCal([[0]])
        restored.load_state_dict(cal.state_dict())
        self.assertEqual(a,restored.predict_one(3000,0,12,20.,2.,(1.,0.),1.))

    def test_array_farcal_matches_reference(self):
        groups=((0,1),(1,0)); records=[]
        for sensor in range(2):
            records.extend(Residual(f"{sensor}:{i}",sensor,3,-59+i,-59+i,float(i%9+1),(sensor*.2,i%3*.1)) for i in range(60))
        reference=FARCal(groups); reference.initialize(records)
        intervals,_=reference.predict_one(1,0,3,20.,2.,(.1,.1),.25)
        arrays={"score":np.asarray([r.score for r in records]),"target":np.asarray([r.target_bin for r in records]),
                "sensor":np.asarray([r.sensor_id for r in records]),"context":np.asarray([r.context for r in records])}
        parameters={"tau":288.,"bandwidth":1.,"support":50.,"stale_decay":288.,"beta_gap":.25,"beta_missing":.25,"inflation_cap":3.}
        factors=_method_widths("far_cal",arrays,arrays,1,0,3,np.asarray((.1,.1)),.25,groups,parameters)
        self.assertTrue(np.allclose([row[2]-20. for row in intervals],2*np.asarray(factors)))

    def test_tuning_archive_uses_only_warmup_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"chunk.npz"
            q50=np.zeros((2,3,1)); q05=q50-1; q95=q50+1; truth=np.asarray([[[1.],[1.],[1.]],[[9.],[9.],[9.]]])
            np.savez_compressed(path,issue_bin=np.asarray([0,96]),episode_index=np.asarray([0,1]),q05_mph=q05,
                q50_mph=q50,q95_mph=q95,truth_mph=truth,original_valid=np.ones((2,3,1),bool),context=np.zeros((2,1,2)))
            manifest={"chunks":[{"file":path.name,"sha256":sha256_file(path)}]}
            records,base=_calibration_records(directory,manifest,(3,),episode_subset=[0])
            self.assertEqual(base,288)
            np.testing.assert_array_equal(records[3]["score"],[1.])

    def test_hidden_valid_truth_stays_in_denominator(self):
        out=interval_metrics([10.,100.,float("nan")],[10.,10.,0.],[5.,5.,0.],[15.,15.,0.],[True,True,False],.1)
        self.assertEqual(out["count"],2)
        self.assertEqual(out["picp"],.5)
        self.assertEqual(out["interval_score"],860.)


if __name__ == "__main__":
    unittest.main()
