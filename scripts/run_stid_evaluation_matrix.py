"""Build STID caches and run the registered real-data C3/C4 audit."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ps1.config import resolve
from ps1.evaluation.farcal_online import evaluate_continuous_farcal, summarize_real_transfer
from ps1.evaluation.online_compare import evaluate_received_online
from ps1.prediction_cache import cache_clean_predictions, cache_faulted_predictions
from ps1.utils.artifacts import write_json


def write_progress(path: Path, payload: dict):
    """Atomically replace mutable evaluation telemetry."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.root.resolve(); artifacts = root / "artifacts"
    locked = json.loads((root / "configs/sumo_farcal_asymmetric_locked.json").read_text(encoding="utf-8"))
    runs = []; started = time.perf_counter(); total = 2 * 3 * (1 + 2 * 3 * 3)
    progress_path = artifacts / "stid-evaluation-progress.json"
    for dataset in ("metr_la", "pems_bay"):
        settings = resolve(root / "configs/base.yaml", root / f"configs/{dataset}.yaml")["calibration"]
        parameters = {"tau": settings["age_decay_bins"], "bandwidth": settings["context_bandwidth"],
            "support": settings["support_reference"], "stale_decay": settings["stale_decay_bins"],
            "beta_gap": settings["beta_gap"], "beta_missing": settings["beta_missing"],
            "inflation_cap": settings["inflation_cap"], "buffer_bins": settings["buffer_days"] * 288,
            **locked["parameters"], "scalable_local_only": True, "records_per_sensor_cap": 256}
        prepared = root / f"data/prepared/{dataset}.npz"
        batch_size = 64
        evaluator = {"prepared_path": prepared, "parameters": parameters,
            "candidate_id": locked["selected_candidate"], "evaluation_stride": 12,
            "focus_fault_events": True, "focus_recovery_bins": 12}
        for model_seed in (11, 22, 33):
            run_name = f"stid-{dataset}-seed{model_seed}"
            checkpoint = artifacts / "training" / run_name / "checkpoint.pt"
            if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
            prediction = artifacts / "predictions" / run_name
            cache_clean_predictions(prepared, checkpoint, prediction / "calibration", "calibration",
                                    batch_size=batch_size, device=args.device)
            runs.append({"run": f"{run_name}-calibration", "status": "completed"})
            for scenario in ("C3", "C4"):
                for fault_seed in (101, 202, 303):
                    fault = artifacts / "faults" / dataset / f"{scenario}-seed{fault_seed}"
                    test = prediction / f"{scenario}-fault{fault_seed}"
                    cache_faulted_predictions(prepared, checkpoint, fault, test,
                                              batch_size=batch_size, device=args.device)
                    online = artifacts / "online" / f"{run_name}-{scenario}-feedback-{scenario}-fault{fault_seed}"
                    evaluate_received_online(prediction / "calibration", test, fault, fault, online)
                    farcal = artifacts / "farcal" / f"{run_name}-{scenario}-fault{fault_seed}-far_cal_asymmetric-asym100"
                    evaluate_continuous_farcal(prediction / "calibration", test, fault, farcal, evaluator)
                    runs.extend({"run": f"{run_name}-{scenario}-fault{fault_seed}-{stage}", "status": "completed"}
                                for stage in ("cache", "online", "farcal"))
                    write_progress(progress_path, {"status": "running", "completed": len(runs), "total": total,
                        "last_run": runs[-1]["run"], "runtime_seconds": time.perf_counter() - started})
    summaries = []
    for dataset in ("metr_la", "pems_bay"):
        for scenario in ("C3", "C4"):
            output = artifacts / "evaluation" / f"stid-{dataset}-asymmetric-{scenario}-transfer-audit.json"
            summaries.append(summarize_real_transfer(artifacts / "farcal", artifacts / "online",
                artifacts / "predictions", output, dataset=dataset, scenario=scenario,
                artifact_prefix="stid-"))
    result = {"status": "completed", "runs": runs, "summaries": [{"dataset": x["dataset"],
        "scenario": x["scenario"], "superiority_rule_met": x["superiority_rule_met"]} for x in summaries],
        "runtime_seconds": time.perf_counter() - started}
    write_json(artifacts / "stid-evaluation-matrix.json", result); print(json.dumps(result))


if __name__ == "__main__": main()

