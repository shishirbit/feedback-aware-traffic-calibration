"""Resume-aware matched-rate independent C3/C4 control evaluation."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ps1.config import config_hash, resolve
from ps1.evaluation.farcal_online import evaluate_continuous_farcal, summarize_real_transfer
from ps1.evaluation.online_compare import evaluate_received_online
from ps1.online.faults import FaultSchedule
from ps1.online.stress import matched_independent_control
from ps1.prediction_cache import cache_clean_predictions, cache_faulted_predictions
from ps1.utils.artifacts import sha256_file, write_json
from run_real_core_evaluation_matrix import completed_manifest, write_progress


def prepare_control(source_dir: Path, destination: Path, control_seed: int) -> dict:
    source_path = source_dir / "fault_manifest.json"
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    with np.load(source_dir / "release_schedule.npz") as data:
        original = FaultSchedule(source_manifest["scenario_id"], source_manifest["seed"],
                                 data["input_arrival"].copy(), data["feedback_arrival"].copy(),
                                 tuple(source_manifest["events"]))
    control = matched_independent_control(original, control_seed)
    destination.mkdir(parents=True, exist_ok=True)
    schedule_path = destination / "release_schedule.npz"
    if schedule_path.exists():
        with np.load(schedule_path) as data:
            if not (np.array_equal(data["input_arrival"], control.input_arrival) and
                    np.array_equal(data["feedback_arrival"], control.feedback_arrival)):
                raise FileExistsError(f"control schedule differs: {schedule_path}")
    else:
        np.savez_compressed(schedule_path, input_arrival=control.input_arrival,
                            feedback_arrival=control.feedback_arrival)
    manifest = control.manifest()
    manifest.update({key: source_manifest[key] for key in ("dataset", "test_start_row", "test_length")})
    manifest.update({
        "control_role": "matched_independent_per_bin_sensor_permutation",
        "control_random_seed": control_seed,
        "source_fault_manifest_sha256": sha256_file(source_path),
        "source_release_schedule_sha256": sha256_file(source_dir / "release_schedule.npz"),
        "release_schedule_sha256": sha256_file(schedule_path),
        "matching_rule": "same windows, failed-entry counts and delay/loss draws in every bin",
    })
    manifest_path = destination / "fault_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old != manifest:
            raise FileExistsError(f"control manifest differs: {manifest_path}")
    else:
        write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    artifacts = root / "artifacts"
    stage = artifacts / "stress" / "matched_control"
    registry_path = root / "configs/real_stress_matched_control.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    locked = json.loads((root / "configs/sumo_farcal_asymmetric_locked.json").read_text(encoding="utf-8"))
    datasets, scenarios = registry["datasets"], registry["scenarios"]
    model_seeds, fault_seeds = registry["model_seeds"], registry["fault_seeds"]
    progress_path = stage / "progress.json"
    stage.mkdir(parents=True, exist_ok=True)
    runs = []
    total = len(datasets) * len(scenarios) * len(fault_seeds) * (1 + 3 * len(model_seeds))
    started = time.perf_counter()

    def record(name: str, status: str) -> None:
        runs.append({"run": name, "status": status})
        write_progress(progress_path, {"status": "running", "completed": len(runs),
            "total": total, "last_run": name, "last_stage_status": status,
            "runtime_seconds": time.perf_counter() - started,
            "registry_sha256": sha256_file(registry_path)})

    for dataset in datasets:
        for scenario in scenarios:
            for fault_seed in fault_seeds:
                name = f"{dataset}-{scenario}-fault{fault_seed}"
                source = artifacts / "faults" / dataset / f"{scenario}-seed{fault_seed}"
                fault = stage / "faults" / dataset / f"{scenario}-seed{fault_seed}"
                existed = (fault / "fault_manifest.json").is_file()
                prepare_control(source, fault, registry["control_random_seed_offset"] + fault_seed)
                record(name + "-schedule", "existing" if existed else "completed")
    if args.prepare_only:
        print(json.dumps({"prepared_schedules": len(runs), "total_schedules":
                          len(datasets) * len(scenarios) * len(fault_seeds)}))
        return

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
        evaluator = {"prepared_path": prepared, "parameters": parameters,
            "candidate_id": locked["selected_candidate"],
            "evaluation_stride": registry["evaluation_stride_bins"],
            "focus_fault_events": True,
            "focus_recovery_bins": registry["focus_recovery_bins"]}
        batch_size = 16 if dataset == "metr_la" else 8
        for model_seed in model_seeds:
            run_name = f"{dataset}-seed{model_seed}"
            checkpoint = artifacts / "training" / f"{run_name}-{config_hash(config)[:12]}" / "checkpoint.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            calibration = artifacts / "predictions" / run_name / "calibration"
            cache_clean_predictions(prepared, checkpoint, calibration, "calibration",
                                    batch_size=batch_size, device=args.device)
            for scenario in scenarios:
                for fault_seed in fault_seeds:
                    identity = f"{run_name}-{scenario}-fault{fault_seed}"
                    fault = stage / "faults" / dataset / f"{scenario}-seed{fault_seed}"
                    test = stage / "predictions" / run_name / f"{scenario}-fault{fault_seed}"
                    existed = completed_manifest(test)
                    cache_faulted_predictions(prepared, checkpoint, fault, test,
                                              batch_size=batch_size, device=args.device)
                    record(identity + "-cache", "existing" if existed else "completed")
                    online = stage / "online" / f"{run_name}-{scenario}-feedback-{scenario}-fault{fault_seed}"
                    existed = completed_manifest(online)
                    if not existed:
                        evaluate_received_online(calibration, test, fault, fault, online)
                    record(identity + "-online", "existing" if existed else "completed")
                    farcal = stage / "farcal" / f"{identity}-far_cal_asymmetric-asym100"
                    existed = completed_manifest(farcal)
                    if not existed:
                        evaluate_continuous_farcal(calibration, test, fault, farcal, evaluator)
                    record(identity + "-farcal", "existing" if existed else "completed")
    summaries = []
    for dataset in datasets:
        for scenario in scenarios:
            output = stage / "evaluation" / f"{dataset}-{scenario}-transfer-audit.json"
            summary = summarize_real_transfer(stage / "farcal", stage / "online",
                stage / "predictions", output, dataset=dataset, scenario=scenario,
                model_seeds=model_seeds, fault_seeds=fault_seeds,
                resamples=registry["bootstrap_resamples"])
            summaries.append({"dataset": dataset, "scenario": scenario,
                "superiority_rule_met": summary["superiority_rule_met"],
                "far_cal_coverage": summary["far_cal_coverage"]})
    result = {"schema_version": 1, "status": "completed", "completed": total,
        "total": total, "registry_sha256": sha256_file(registry_path),
        "runs": runs, "summaries": summaries,
        "scope_boundary": registry["scope_boundary"]}
    write_json(stage / "matrix.json", result)
    write_progress(progress_path, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
