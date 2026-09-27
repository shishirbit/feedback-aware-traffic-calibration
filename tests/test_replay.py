import unittest
from ps1.online.events import Observation
from ps1.online.gateway import ReportingGateway
from ps1.online.history import InputHistory
from ps1.online.replay import OnlineEngine
from ps1.models.baselines import LastAvailable
from ps1.calibration.far_cal import FARCal, Residual


def run(changed_after=999, resume_at=None):
    cal=FARCal([[0]])
    cal.initialize([Residual(f"archive:{h}:{t}",0,h,t,t,1.,(1.,0.)) for h in range(1,13) for t in range(-55,-5)])
    engine=OnlineEngine(InputHistory("test",[40.]),LastAvailable([2.]),cal)
    gateway=ReportingGateway(lambda o,c:o.observed_bin+3)
    for now in range(30):
        gateway.ingest_truth(Observation("test",0,now,40.+now if now<changed_after else 999.))
        engine.step(now,gateway.release_due(now),issue=11<=now<18)
        if now==resume_at:
            engine=OnlineEngine.restore(engine.checkpoint())
    return engine


class ReplayTests(unittest.TestCase):
    def test_resume_matches_uninterrupted(self):
        full,restored=run(),run(resume_at=14)
        self.assertEqual(full.ledger.records,restored.ledger.records)
        self.assertEqual(full.ledger.feedback_matches,restored.ledger.feedback_matches)
        self.assertEqual(full.diagnostics,restored.diagnostics)

    def test_future_truth_and_unreleased_truth_cannot_change_forecasts(self):
        full,changed=run(),run(changed_after=14)
        for fid,record in full.ledger.records.items():
            if record.issue_bin<17:
                self.assertEqual(record,changed.ledger.records[fid])

    def test_outcomes_are_never_received_before_target_plus_delay(self):
        engine=run()
        for fid,event,arrival,target,residual,miss in engine.ledger.feedback_matches:
            self.assertGreaterEqual(arrival,target+3)


if __name__=="__main__":
    unittest.main()
