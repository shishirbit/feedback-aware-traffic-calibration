"""Run the locked FAR-GW C1/C2/C5 real-data core matrix."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ps1.config import config_hash, resolve
from ps1.evaluation.farcal_online import evaluate_continuous_farcal, summarize_real_transfer
from ps1.evaluation.online_compare import evaluate_received_online
from ps1.prediction_cache import cache_clean_predictions, cache_faulted_predictions
from ps1.utils.artifacts import sha256_file, write_json


def write_progress(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def completed_manifest(directory: Path) -> bool:
    path = directory / "manifest.json"
    if not path.is_file():
        return False
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        return False
    for chunk in manifest.get("chunks", []):
        chunk_path = directory / chunk["file"]
        if not chunk_path.is_file() or sha256_file(chunk_path) != chunk["sha256"]:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.root.resolve()
    artifacts = root / "artifacts"
    registry_path = root / "configs/real_core_c1_c2_c5.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    locked = json.loads((root / "configs/sumo_farcal_asymmetric_locked.json").read_text(encoding="utf-8"))
    datasets = tuple(registry["datasets"])
    scenarios = tuple(registry["scenarios"])
    model_seeds = tuple(registry["model_seeds"])
    fault_seeds = tuple(registry["fault_seeds"])
    total = len(datasets) * len(model_seeds) * (1 + len(scenarios) * len(fault_seeds) * 3)
    progress_path = artifacts / "real-core-evaluation-progress.json"
    runs: list[dict[str, object]] = []
    started = time.perf_counter()

    def record(run: str, status: str) -> None:
        runs.append({"run": run, "status": status})
        write_progress(progress_path, {
            "status": "running", "completed": len(runs), "total": total,
            "last_run": run, "last_stage_status": status,
            "runtime_seconds": time.perf_counter() - started,
            "registry_sha256": sha256_file(registry_path),
        })

    for dataset in datasets:
        config = resolve(root / "configs/base.yaml", root / f"configs/{dataset}.yaml")
        settings = config["calibration"]
        parameters = {
            "tau": settings["age_decay_bins"], "bandwidth": settings["context_bandwidth"],
            "support": settings["support_reference"], "stale_decay": settings["stale_decay_bins"],
            "beta_gap": settings["beta_gap"], "beta_missing": settings["beta_missing"],
            "inflation_cap": settings["inflation_cap"], "buffer_bins": settings["buffer_days"] * 288,
            **locked["parameters"], "scalable_local_only": True, "records_per_sensor_cap": 256,
        }
        prepared = root / f"data/prepared/{dataset}.npz"
        evaluator = {
            "prepared_path": prepared, "parameters": parameters,
            "candidate_id": locked["selected_candidate"],
            "evaluation_stride": registry["evaluation_stride_bins"],
            "focus_fault_events": registry["focus_fault_events"],
            "focus_recovery_bins": registry["focus_recovery_bins"],
        }
        # Prediction caching is inference-only, so use conservative GPU batches.
        # The training batch of 32 triggered a transient cuDNN execution failure
        # after extended replay on this 8 GB laptop GPU.
        batch_size = 16 if dataset == "metr_la" else 8
        for model_seed in model_seeds:
            run_name = f"{dataset}-seed{model_seed}"
            checkpoint = artifacts / "training" / f"{run_name}-{config_hash(config)[:12]}" / "checkpoint.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            prediction = artifacts / "predictions" / run_name
            existed = completed_manifest(prediction / "calibration")
            cache_clean_predictions(prepared, checkpoint, prediction / "calibration", "calibration",
                                    batch_size=batch_size, device=args.device)
            record(f"{run_name}-calibration", "existing" if existed else "completed")
            for scenario in scenarios:
                for fault_seed in fault_seeds:
                    identity = f"{run_name}-{scenario}-fault{fault_seed}"
                    fault = artifacts / "faults" / dataset / f"{scenario}-seed{fault_seed}"
                    test = prediction / f"{scenario}-fault{fault_seed}"
                    existed = completed_manifest(test)
                    cache_faulted_predictions(prepared, checkpoint, fault, test,
                                              batch_size=batch_size, device=args.device)
                    record(f"{identity}-cache", "existing" if existed else "completed")

                    online = artifacts / "online" / f"{run_name}-{scenario}-feedback-{scenario}-fault{fault_seed}"
                    existed = completed_manifest(online)
                    if not existed:
                        evaluate_received_online(prediction / "calibration", test, fault, fault, online)
                    record(f"{identity}-online", "existing" if existed else "completed")

                    farcal = artifacts / "farcal" / f"{identity}-far_cal_asymmetric-asym100"
                    existed = completed_manifest(farcal)
                    if not existed:
                        scenario_evaluator = dict(evaluator)
                        # C5 is independent loss over the full stream and has no
                        # bounded outage event window. Evaluate its registered
                        # full stream rather than inventing a spatial focus set.
                        if scenario == "C5":
                            scenario_evaluator["focus_fault_events"] = False
                            scenario_evaluator["focus_recovery_bins"] = 0
                        evaluate_continuous_farcal(prediction / "calibration", test, fault, farcal,
                                                   scenario_evaluator)
                    record(f"{identity}-farcal", "existing" if existed else "completed")

    summaries = []
    for dataset in datasets:
        for scenario in scenarios:
            output = artifacts / "evaluation" / f"{dataset}-asymmetric-{scenario}-transfer-audit.json"
            summary = summarize_real_transfer(
                artifacts / "farcal", artifacts / "online", artifacts / "predictions", output,
                dataset=dataset, model_seeds=model_seeds, fault_seeds=fault_seeds,
                resamples=registry["bootstrap_resamples"], scenario=scenario,
            )
            summaries.append({"dataset": dataset, "scenario": scenario,
                "far_cal_coverage": summary["far_cal_coverage"],
                "superiority_rule_met": summary["superiority_rule_met"]})

    result = {
        "schema_version": 1, "status": "completed",
        "registry_path": str(registry_path.relative_to(root)),
        "registry_sha256": sha256_file(registry_path), "runs": runs, "summaries": summaries,
        "runtime_seconds": time.perf_counter() - started, "scope_boundary": registry["scope_boundary"],
    }
    write_json(artifacts / "real-core-c1-c2-c5-matrix.json", result)
    write_progress(progress_path, {**result, "completed": total, "total": total})
    print(json.dumps(result))


if __name__ == "__main__":
    main()
