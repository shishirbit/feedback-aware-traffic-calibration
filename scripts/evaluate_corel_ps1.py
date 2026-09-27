"""Evaluate the trained PS-1 CoRel adapter with causal feedback arrivals."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from run_corel_ps1 import COREL_COMMIT, QUANTILES, build_model, load_cache, sha256


REPORT_HORIZONS = (3, 6, 12)


def residuals_by_target(arrays: dict[str, np.ndarray], start: int, length: int):
    origins, horizons, sensors = arrays["truth_mph"].shape
    values = np.zeros((length, sensors, horizons), dtype=np.float32)
    valid = np.zeros_like(values, dtype=bool)
    errors = arrays["truth_mph"] - arrays["q50_mph"]
    source_valid = arrays["original_valid"] & np.isfinite(errors)
    for row, issue in enumerate(arrays["issue_bin"]):
        for h_idx in range(horizons):
            target = int(issue) + h_idx + 1 - start
            if 0 <= target < length:
                values[target, :, h_idx] = errors[row, h_idx]
                valid[target, :, h_idx] = source_valid[row, h_idx]
    return values, valid


def initial_history(calibration: dict[str, np.ndarray], first_issue: int, window: int = 12):
    first_target = int(calibration["issue_bin"][0]) + 1
    length = first_issue - first_target
    values, valid = residuals_by_target(calibration, first_target, length)
    state = np.zeros(values.shape[1:], dtype=np.float32)
    history: deque[np.ndarray] = deque(maxlen=window)
    for row in range(length):
        state[valid[row]] = values[row][valid[row]]
        history.append(state.copy())
    if len(history) < window:
        raise ValueError("insufficient calibration history")
    return history, state


def interval_sums(truth, lower, upper, mask, alpha=0.1):
    y, lo, up = truth[mask], lower[mask], upper[mask]
    score = up - lo + 2 / alpha * np.maximum(lo - y, 0) + 2 / alpha * np.maximum(y - up, 0)
    return {"count": int(len(y)), "coverage_sum": float(((y >= lo) & (y <= up)).sum()),
            "width_sum": float((up - lo).sum()), "interval_score_sum": float(score.sum())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--fault-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corel-source", type=Path, default=Path("third_party/corel"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--recovery-bins", type=int, default=12)
    args = parser.parse_args()

    source = args.corel_source.resolve()
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != COREL_COMMIT:
        raise RuntimeError(f"expected CoRel {COREL_COMMIT}, found {commit}")
    calibration_manifest, calibration = load_cache(args.calibration_cache)
    test_manifest, test = load_cache(args.test_cache)
    fault_path = args.fault_dir / "fault_manifest.json"
    schedule_path = args.fault_dir / "release_schedule.npz"
    fault = json.loads(fault_path.read_text(encoding="utf-8"))
    if test_manifest["release_schedule_sha256"] != sha256(schedule_path):
        raise ValueError("test cache and release schedule differ")
    if calibration_manifest["checkpoint_sha256"] != test_manifest["checkpoint_sha256"]:
        raise ValueError("calibration and test point checkpoints differ")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["source_commit"] != commit or checkpoint["model_seed"] != test_manifest["checkpoint_training_seed"]:
        raise ValueError("CoRel checkpoint identity mismatch")
    sensors, horizons = test["q50_mph"].shape[2], test["q50_mph"].shape[1]
    model = build_model(source, sensors, horizons)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    means, stds = np.asarray(checkpoint["mean"]), np.asarray(checkpoint["std"])
    q_low = int(np.flatnonzero(np.isclose(QUANTILES, 0.05))[0])
    q_high = int(np.flatnonzero(np.isclose(QUANTILES, 0.95))[0])
    torch.set_num_threads(1)

    test_start = int(fault["test_start_row"])
    with np.load(schedule_path) as schedule:
        feedback_arrival = schedule["feedback_arrival"].copy()
    target_values, target_valid = residuals_by_target(test, test_start, len(feedback_arrival))
    observed, event_sensor = np.nonzero(feedback_arrival >= 0)
    arrival = feedback_arrival[observed, event_sensor]
    order = np.argsort(arrival, kind="stable")
    arrival, observed, event_sensor = arrival[order], observed[order], event_sensor[order]
    pointer = 0

    issue_bins = test["issue_bin"]
    history, state = initial_history(calibration, int(issue_bins[0]))
    selected = np.arange(0, len(issue_bins), args.stride)
    selected_lookup = {int(row): out for out, row in enumerate(selected)}
    lower = np.full((len(selected), 1, 1, len(REPORT_HORIZONS), sensors), np.nan, np.float32)
    upper = np.full_like(lower, np.nan)
    evaluation_mask = np.zeros((len(selected), sensors), dtype=bool)
    output_issue = issue_bins[selected].copy()
    started = time.perf_counter()

    for row, issue in enumerate(issue_bins):
        now = int(issue)
        local = now - test_start
        out_row = selected_lookup.get(row)
        if out_row is not None:
            x = np.stack(history)
            x = (x - means[None, None, :]) / stds[None, None, :]
            with torch.no_grad():
                prediction = model(torch.from_numpy(x[None].astype(np.float32)))[:, 0, 0].cpu().numpy()
            raw = prediction * stds[None, None, :] + means[None, None, :]
            focus = set()
            for event in fault["events"]:
                if int(event["start_bin"]) <= local <= int(event["end_bin"]) + args.recovery_bins:
                    focus.update(event["affected_sensors"])
            if focus:
                evaluation_mask[out_row, sorted(focus)] = True
            for hi, horizon in enumerate(REPORT_HORIZONS):
                center = test["q50_mph"][row, horizon - 1]
                lo = np.maximum(0, center + raw[q_low, :, horizon - 1])
                up = np.maximum(lo, center + raw[q_high, :, horizon - 1])
                lower[out_row, 0, 0, hi] = lo
                upper[out_row, 0, 0, hi] = up

        # Match the training windows: outcomes released at bin t enter the
        # residual history used from t+1 onward.
        end = np.searchsorted(arrival, local, side="right")
        for index in range(pointer, end):
            target, sensor = int(observed[index]), int(event_sensor[index])
            mask = target_valid[target, sensor]
            state[sensor, mask] = target_values[target, sensor, mask]
        pointer = end
        history.append(state.copy())

    truth = test["truth_mph"][selected][:, np.asarray(REPORT_HORIZONS) - 1]
    valid = test["original_valid"][selected][:, np.asarray(REPORT_HORIZONS) - 1]
    metrics = {}
    for hi, horizon in enumerate(REPORT_HORIZONS):
        sums = interval_sums(truth[:, hi], lower[:, 0, 0, hi], upper[:, 0, 0, hi],
                             valid[:, hi] & evaluation_mask)
        count = sums["count"]
        metrics[str(horizon)] = {**sums, "coverage": sums["coverage_sum"] / count,
                                 "mean_width": sums["width_sum"] / count,
                                 "mean_interval_score": sums["interval_score_sum"] / count}

    args.output.mkdir(parents=True, exist_ok=True)
    chunk = args.output / "chunk-0000.npz"
    np.savez_compressed(chunk, issue_bin=output_issue, lower_mph=lower, upper_mph=upper,
                        evaluation_mask=evaluation_mask)
    result = {
        "schema_version": 1, "status": "completed", "method": "corel_ps1_adapter",
        "dataset": test_manifest["dataset"], "model_seed": test_manifest["checkpoint_training_seed"],
        "scenario": fault["scenario_id"], "fault_seed": fault["seed"], "evaluation_stride": args.stride,
        "focus_recovery_bins": args.recovery_bins, "report_horizons": list(REPORT_HORIZONS),
        "levels": [0.9], "metrics": metrics, "source_commit": commit,
        "checkpoint_sha256": sha256(args.checkpoint),
        "calibration_manifest_sha256": sha256(args.calibration_cache / "manifest.json"),
        "test_manifest_sha256": sha256(args.test_cache / "manifest.json"),
        "fault_manifest_sha256": sha256(fault_path), "release_schedule_sha256": sha256(schedule_path),
        "chunks": [{"file": chunk.name, "sha256": sha256(chunk), "origins": len(selected)}],
        "feedback_timing": "release at bin t enters CoRel history for forecasts from t+1",
        "runtime_seconds": time.perf_counter() - started,
    }
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "dataset": result["dataset"],
                      "scenario": result["scenario"], "origins": len(selected),
                      "runtime_seconds": result["runtime_seconds"]}))


if __name__ == "__main__":
    main()
