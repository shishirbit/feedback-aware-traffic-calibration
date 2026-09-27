"""Resumable locked C3/C4 one-factor real-data stress matrix."""
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
from ps1.online.faults import FaultSchedule, generate_fault_schedule
from ps1.online.stress import loss_variant_from_source, mnar_schedule, one_factor_schedule
from ps1.prediction_cache import cache_clean_predictions, cache_faulted_predictions
from ps1.utils.artifacts import sha256_file, write_json
from run_real_core_evaluation_matrix import completed_manifest, write_progress


def variants(scenario: str, registry: dict) -> list[tuple[str, str, float | None]]:
    values = registry["one_factor_values"]
    result = []
    for value in values["duration_bins"]:
        result.append((f"duration{value:02d}", "duration_bins", value))
    for value in values["connected_fraction"]:
        result.append((f"fraction{round(100 * value):02d}", "connected_fraction", value))
    for value in values["additional_backlog_delay_bins"]:
        result.append((f"extra{value:02d}", "additional_backlog_delay_bins", value))
    baseline_loss = registry["baseline"][f"{scenario}_loss_within"]
    for value in values["loss_within"]:
        if value != baseline_loss:
            result.append((f"loss{round(100 * value):03d}", "loss_within", value))
    result.extend((("cold_calibration", "cold_calibration", None),
                   ("mnar_low_speed", "mnar_low_speed", None)))
    if len(result) != 12:
        raise ValueError("expected twelve distinct stress variants per scenario")
    return result


def read_baseline(source: Path) -> tuple[FaultSchedule, dict]:
    manifest = json.loads((source / "fault_manifest.json").read_text(encoding="utf-8"))
    with np.load(source / "release_schedule.npz") as data:
        schedule = FaultSchedule(manifest["scenario_id"], manifest["seed"],
            data["input_arrival"].copy(), data["feedback_arrival"].copy(), tuple(manifest["events"]))
    return schedule, manifest


def save_schedule(schedule: FaultSchedule, output: Path, metadata: dict) -> bool:
    output.mkdir(parents=True, exist_ok=True)
    schedule_path = output / "release_schedule.npz"
    existed = schedule_path.exists()
    if existed:
        with np.load(schedule_path) as old:
            if not (np.array_equal(old["input_arrival"], schedule.input_arrival) and
                    np.array_equal(old["feedback_arrival"], schedule.feedback_arrival)):
                raise FileExistsError(f"stress schedule differs: {schedule_path}")
    else:
        np.savez_compressed(schedule_path, input_arrival=schedule.input_arrival,
                            feedback_arrival=schedule.feedback_arrival)
    manifest = schedule.manifest()
    manifest.update(metadata)
    manifest["release_schedule_sha256"] = sha256_file(schedule_path)
    write_json(output / "fault_manifest.json", manifest)
    return existed


def training_thresholds(prepared: Path, quantile: float) -> tuple[np.ndarray, np.ndarray, int, int, np.ndarray]:
    with np.load(prepared) as data:
        values = data["values_mph"].copy()
        valid = data["original_valid"].copy()
        times = data["timestamp_ns"].copy()
        adjacency = data["adjacency"].copy()
    boundary = json.loads(prepared.with_suffix(".manifest.json").read_text(encoding="utf-8"))["boundary_timestamps"]
    indices = [int(np.searchsorted(times, np.datetime64(stamp).astype("datetime64[ns]").astype("int64")))
               for stamp in boundary]
    training = np.where(valid[indices[0]:indices[1]], values[indices[0]:indices[1]], np.nan)
    thresholds = np.nanquantile(training, quantile, axis=0)
    if not np.isfinite(thresholds).all():
        raise ValueError("training-derived low-speed thresholds are nonfinite")
    return values, thresholds, indices[2], indices[3], adjacency


def prepare_variant(root: Path, stage: Path, registry: dict, dataset: str, scenario: str,
                    fault_seed: int, variant: tuple[str, str, float | None],
                    prepared_data: tuple) -> tuple[Path, Path | None, bool, bool | None]:
    variant_id, factor, value = variant
    prepared = root / f"data/prepared/{dataset}.npz"
    values, thresholds, calibration_start, test_start, adjacency = prepared_data
    source = root / "artifacts/faults" / dataset / f"{scenario}-seed{fault_seed}"
    baseline, source_manifest = read_baseline(source)
    if factor in {"cold_calibration", "mnar_low_speed"}:
        if factor == "cold_calibration":
            schedule = baseline
        else:
            schedule = mnar_schedule(baseline, values, thresholds, test_start,
                registry["mnar"]["random_seed_offset"] + fault_seed,
                registry["mnar"]["additional_low_speed_loss_probability"])
    elif factor == "loss_within":
        schedule = loss_variant_from_source(baseline, fault_seed, float(value))
    else:
        schedule = one_factor_schedule(scenario, source_manifest["test_length"], adjacency,
                                       fault_seed, factor, value)
    output = stage / variant_id / "faults" / dataset / f"{scenario}-seed{fault_seed}"
    metadata = {"dataset": dataset, "test_start_row": test_start,
        "test_length": source_manifest["test_length"], "variant": variant_id,
        "one_factor": factor, "factor_value": value,
        "source_fault_manifest_sha256": sha256_file(source / "fault_manifest.json"),
        "training_quantile": registry["mnar"]["low_speed_training_quantile"] if factor == "mnar_low_speed" else None,
        "training_low_speed_threshold_mph": thresholds.tolist() if factor == "mnar_low_speed" else None,
        "true_regime_visible_to_model": False}
    existed = save_schedule(schedule, output, metadata)
    calibration_fault = None
    if factor == "cold_calibration":
        # The calibration partition immediately precedes test. It receives its
        # own seeded C3/C4 process; the test process remains unchanged.
        calibration_length = test_start - calibration_start
        cal_schedule = generate_fault_schedule(scenario, calibration_length, adjacency,
            3000000 + fault_seed, connected_fraction=.10, duration_bins=6,
            backlog_release_bins=3, events_per_day=2, loss_within=.50)
        calibration_fault = stage / variant_id / "calibration_faults" / dataset / f"{scenario}-seed{fault_seed}"
        cal_existed=(calibration_fault / "fault_manifest.json").is_file()
        save_schedule(cal_schedule, calibration_fault, {"dataset": dataset,
            "partition": "calibration", "partition_start_row": calibration_start,
            "partition_length": calibration_length, "test_start_row": calibration_start,
            "test_length": calibration_length, "variant": variant_id,
            "reporting_process": "same C3/C4 parameters as test"})
    return output, calibration_fault, existed, (cal_existed if factor == "cold_calibration" else None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    artifacts = root / "artifacts"
    stage = artifacts / "stress" / "one_factor"
    stage.mkdir(parents=True, exist_ok=True)
    registry_path = root / "configs/real_one_factor_stress.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    locked = json.loads((root / "configs/sumo_farcal_asymmetric_locked.json").read_text(encoding="utf-8"))
    datasets, scenarios = registry["datasets"], registry["scenarios"]
    model_seeds, fault_seeds = registry["model_seeds"], registry["fault_seeds"]
    variant_count = sum(len(variants(s, registry)) for s in scenarios)
    cold_count = len(datasets) * len(scenarios) * len(fault_seeds)
    total = (len(datasets) * len(fault_seeds) * variant_count * (1 + 3 * len(model_seeds))
             + cold_count + cold_count * len(model_seeds))
    runs = []
    started = time.perf_counter()
    progress_path = stage / "progress.json"

    def record(name: str, status: str) -> None:
        runs.append({"run": name, "status": status})
        write_progress(progress_path, {"status": "running", "completed": len(runs),
            "total": total, "last_run": name, "last_stage_status": status,
            "runtime_seconds": time.perf_counter() - started,
            "registry_sha256": sha256_file(registry_path)})

    data_by_dataset = {}
    for dataset in datasets:
        prepared = root / f"data/prepared/{dataset}.npz"
        data_by_dataset[dataset] = training_thresholds(prepared, registry["mnar"]["low_speed_training_quantile"])
        for scenario in scenarios:
            for fault_seed in fault_seeds:
                for variant in variants(scenario, registry):
                    _, calibration_fault, existed, cal_existed = prepare_variant(root, stage, registry, dataset,
                        scenario, fault_seed, variant, data_by_dataset[dataset])
                    record(f"{dataset}-{scenario}-fault{fault_seed}-{variant[0]}-schedule",
                           "existing" if existed else "completed")
                    if calibration_fault is not None:
                        record(f"{dataset}-{scenario}-fault{fault_seed}-cold-calibration-schedule",
                               "existing" if cal_existed else "completed")
    if args.prepare_only:
        print(json.dumps({"prepared": len(runs), "total": total}))
        return

    for dataset in datasets:
        config = resolve(root / "configs/base.yaml", root / f"configs/{dataset}.yaml")
        settings = config["calibration"]
        parameters = {"tau": settings["age_decay_bins"], "bandwidth": settings["context_bandwidth"],
            "support": settings["support_reference"], "stale_decay": settings["stale_decay_bins"],
            "beta_gap": settings["beta_gap"], "beta_missing": settings["beta_missing"],
            "inflation_cap": settings["inflation_cap"], "buffer_bins": settings["buffer_days"] * 288,
            **locked["parameters"], "scalable_local_only": True, "records_per_sensor_cap": 256}
        prepared = root / f"data/prepared/{dataset}.npz"
        batch_size = 16 if dataset == "metr_la" else 8
        for model_seed in model_seeds:
            run_name = f"{dataset}-seed{model_seed}"
            checkpoint = artifacts / "training" / f"{run_name}-{config_hash(config)[:12]}" / "checkpoint.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            clean_calibration = artifacts / "predictions" / run_name / "calibration"
            cache_clean_predictions(prepared, checkpoint, clean_calibration, "calibration",
                                    batch_size=batch_size, device=args.device)
            for scenario in scenarios:
                for fault_seed in fault_seeds:
                    for variant_id, factor, _ in variants(scenario, registry):
                        prefix = stage / variant_id
                        identity = f"{run_name}-{scenario}-fault{fault_seed}"
                        fault = prefix / "faults" / dataset / f"{scenario}-seed{fault_seed}"
                        calibration = clean_calibration
                        calibration_fault = None
                        if factor == "cold_calibration":
                            calibration_fault = prefix / "calibration_faults" / dataset / f"{scenario}-seed{fault_seed}"
                            calibration = prefix / "calibration_predictions" / run_name / f"{scenario}-fault{fault_seed}"
                            existed = completed_manifest(calibration)
                            cache_faulted_predictions(prepared, checkpoint, calibration_fault, calibration,
                                batch_size=batch_size, device=args.device, partition="calibration")
                            record(identity + "-cold-calibration-cache", "existing" if existed else "completed")
                        test = prefix / "predictions" / run_name / f"{scenario}-fault{fault_seed}"
                        existed = completed_manifest(test)
                        cache_faulted_predictions(prepared, checkpoint, fault, test,
                                                  batch_size=batch_size, device=args.device)
                        record(identity + "-" + variant_id + "-cache", "existing" if existed else "completed")
                        online = prefix / "online" / f"{run_name}-{scenario}-feedback-{scenario}-fault{fault_seed}"
                        existed = completed_manifest(online)
                        if not existed:
                            evaluate_received_online(calibration, test, fault, fault, online,
                                calibration_fault_dir=calibration_fault)
                        record(identity + "-" + variant_id + "-online", "existing" if existed else "completed")
                        farcal = prefix / "farcal" / f"{identity}-far_cal_asymmetric-asym100"
                        existed = completed_manifest(farcal)
                        if not existed:
                            evaluator = {"prepared_path": prepared, "parameters": parameters,
                                "candidate_id": locked["selected_candidate"],
                                "evaluation_stride": registry["evaluation_stride_bins"],
                                "focus_fault_events": True,
                                "focus_recovery_bins": registry["focus_recovery_bins"]}
                            if calibration_fault is not None:
                                evaluator["calibration_fault_dir"] = calibration_fault
                            evaluate_continuous_farcal(calibration, test, fault, farcal, evaluator)
                        record(identity + "-" + variant_id + "-farcal", "existing" if existed else "completed")

    summaries = []
    for dataset in datasets:
        for scenario in scenarios:
            for variant_id, factor, value in variants(scenario, registry):
                prefix = stage / variant_id
                output = prefix / "evaluation" / f"{dataset}-{scenario}-transfer-audit.json"
                summary = summarize_real_transfer(prefix / "farcal", prefix / "online", prefix / "predictions",
                    output, dataset=dataset, scenario=scenario, model_seeds=model_seeds,
                    fault_seeds=fault_seeds, resamples=registry["bootstrap_resamples"])
                summaries.append({"dataset": dataset, "scenario": scenario, "variant": variant_id,
                    "factor": factor, "value": value,
                    "superiority_rule_met": summary["superiority_rule_met"],
                    "far_cal_coverage": summary["far_cal_coverage"]})
    result = {"schema_version": 1, "status": "completed", "completed": total,
        "total": total, "registry_sha256": sha256_file(registry_path),
        "summaries": summaries, "scope_boundary": registry["scope_boundary"]}
    write_json(stage / "matrix.json", result)
    write_progress(progress_path, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
