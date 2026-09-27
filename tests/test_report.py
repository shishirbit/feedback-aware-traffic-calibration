import tempfile
import unittest
import json
from pathlib import Path
from ps1.evaluation.report import load_registry,render_status_report,render_clean_report,render_feedback_report


class ReportTests(unittest.TestCase):
    def test_completed_entry_requires_real_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); registry=root/"registry.yaml"
            registry.write_text("schema_version: 1\nexperiments:\n  - id: x\n    status: completed\n    artifact_path: absent\n",encoding="utf-8")
            with self.assertRaises(FileNotFoundError): load_registry(registry,root)

    def test_report_preserves_registry_claim_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/"artifact").write_text("evidence",encoding="utf-8")
            registry=root/"registry.yaml"
            registry.write_text("schema_version: 1\nexperiments:\n  - id: smoke\n    dataset: fixture\n    status: completed\n    artifact_path: artifact\n  - id: paper\n    status: planned\n    reason: not run\n",encoding="utf-8")
            output=render_status_report(registry,root,root/"reports")
            text=output.read_text(encoding="utf-8")
            self.assertIn("bounded by the completed registry entries",text)
            self.assertIn("Planned or blocked entries remain outside",text)
            self.assertIn("not run",text)

    def test_clean_report_preserves_scope_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/"evaluation.json"
            source.write_text(json.dumps({"status":"completed_clean_C0_only","dataset":"demo","training_seed":11,"report_horizons":[3],
                "point":{"3":{"count":10,"mae_mph":1.,"rmse_mph":2.}},
                "intervals":{"raw_q05_q95":{"3":{"picp":.9,"mpiw_mph":3.}},"frozen":{"0.9":{"3":{"picp":.91,"mpiw_mph":4.,"mean_interval_score":5.}}},"primary_equal_horizon_mean_frozen_90_interval_score":5.},
                "limitations":["Single seed"]}),encoding="utf-8")
            output=render_clean_report(source,root/"reports"); text=output.read_text(encoding="utf-8")
            self.assertIn("does not evaluate H1-H4",text); self.assertIn("| 3 | 10 |",text)

    def test_feedback_report_marks_preliminary_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/"comparison.json"
            source.write_text(json.dumps({"status":"completed_single_seed_single_fault","dataset":"demo","model_seed":11,"fault_seed":101,
                "point_scenario":"C3","immediate_feedback_scenario":"C2","delayed_feedback_scenario":"C3",
                "comparisons":{"rolling":{"primary_equal_horizon_90_interval_score_delayed_minus_immediate":{"estimate":.1,"lower":.01,"upper":.2}}},
                "limitations":["One seed"]}),encoding="utf-8")
            text=render_feedback_report(source,root/"reports").read_text(encoding="utf-8")
            self.assertIn("preliminary H1 evidence only",text); self.assertIn("0.1000000",text)

    def test_feedback_report_labels_episode_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/"evaluation.json"
            source.write_text(json.dumps({"status":"completed_single_seed_single_fault","dataset":"sumo","model_seed":11,"fault_seed":101,
                "point_scenario":"C3","immediate_feedback_scenario":"C2","delayed_feedback_scenario":"C3","bootstrap_unit":"episode",
                "comparisons":{"rolling":{"primary_equal_horizon_90_interval_score_delayed_minus_immediate":{"estimate":.1,"lower":.01,"upper":.2}}},
                "limitations":["One seed"]}),encoding="utf-8")
            text=render_feedback_report(source,root/"reports").read_text(encoding="utf-8")
            self.assertIn("95% episode-bootstrap interval",text)


if __name__=="__main__": unittest.main()
