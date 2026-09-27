"""Summarize paired CoRel, FAR-Cal, rolling, and ACI real-data audits."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ps1.utils.artifacts import sha256_file
from ps1.evaluation.farcal_online import _stratified_block_ci
from ps1.evaluation.online_compare import (_align_origins, _cache_arrays,
                                           _cache_manifest, _interval_arrays,
                                           moving_block_mean_ci)


def summarize(root: Path, dataset: str, scenario: str, output: Path, resamples: int = 2000):
    model_seeds, fault_seeds = (11, 22, 33), (101, 202, 303)
    effects = defaultdict(lambda: defaultdict(list))
    method_scores = defaultdict(list)
    method_coverages = defaultdict(list)
    by_run, sources = {}, []
    for fault_seed in fault_seeds:
        for model_seed in model_seeds:
            far_dir = root / "farcal" / f"{dataset}-seed{model_seed}-{scenario}-fault{fault_seed}-far_cal_asymmetric-asym100"
            online_dir = root / "online" / f"{dataset}-seed{model_seed}-{scenario}-feedback-{scenario}-fault{fault_seed}"
            test_dir = root / "predictions" / f"{dataset}-seed{model_seed}" / f"{scenario}-fault{fault_seed}"
            corel_dir = root / "corel-evaluation" / f"{dataset}-seed{model_seed}-{scenario}-fault{fault_seed}"
            far, far_path = _cache_manifest(far_dir)
            online, online_path = _cache_manifest(online_dir)
            test, test_path = _cache_manifest(test_dir)
            corel, corel_path = _cache_manifest(corel_dir)
            if "chunks" not in corel and "chunk" in corel:
                corel = {**corel, "chunks": [corel["chunk"]]}
            fa = _cache_arrays(far_dir, far)
            oa = _align_origins(_cache_arrays(online_dir, online), fa["issue_bin"])
            ta = _align_origins(_cache_arrays(test_dir, test), fa["issue_bin"])
            ca = _align_origins(_cache_arrays(corel_dir, corel), fa["issue_bin"])
            if not np.array_equal(fa["evaluation_mask"], ca["evaluation_mask"]):
                raise ValueError(f"evaluation mask mismatch: {corel_dir}")
            horizons = np.asarray(far["report_horizons"])
            truth = ta["truth_mph"][:, horizons - 1]
            mask = ta["original_valid"][:, horizons - 1] & fa["evaluation_mask"][:, None, :]

            def extract(lower, upper):
                pieces = [_interval_arrays(truth[:, hi], lower[:, hi], upper[:, hi], mask[:, hi], .1)
                          for hi in range(len(horizons))]
                return np.stack([part[0] for part in pieces], axis=1), np.stack([part[1] for part in pieces], axis=1)

            fi = far["methods"].index("far_cal_asymmetric")
            far_score, far_cov = extract(fa["lower_mph"][:, fi, 0], fa["upper_mph"][:, fi, 0])
            corel_score, corel_cov = extract(ca["lower_mph"][:, 0, 0], ca["upper_mph"][:, 0, 0])
            scores = {"far_cal": far_score, "corel": corel_score}
            coverages = {"far_cal": far_cov, "corel": corel_cov}
            for name in ("rolling", "aci"):
                index = online["methods"].index(name)
                scores[name], coverages[name] = extract(oa["lower_mph"][:, index, 0], oa["upper_mph"][:, index, 0])
            differences = {
                "corel_minus_far_cal": scores["corel"] - scores["far_cal"],
                "rolling_minus_corel": scores["rolling"] - scores["corel"],
                "aci_minus_corel": scores["aci"] - scores["corel"],
            }
            for name, value in scores.items():
                method_scores[name].append(value)
                method_coverages[name].append(coverages[name])
            for name, value in differences.items():
                effects[name][fault_seed].append(value)
            by_run[f"model{model_seed}-fault{fault_seed}"] = {
                "scores": {name: float(np.nanmean(value)) for name, value in scores.items()},
                "coverage": {name: float(np.nanmean(value)) for name, value in coverages.items()},
                "effects": {name: moving_block_mean_ci(value, 24, resamples, 0) for name, value in differences.items()},
            }
            sources.append({"model_seed": model_seed, "fault_seed": fault_seed,
                            "farcal_manifest_sha256": sha256_file(far_path),
                            "corel_manifest_sha256": sha256_file(corel_path),
                            "online_manifest_sha256": sha256_file(online_path),
                            "test_manifest_sha256": sha256_file(test_path)})

    primary = {}
    for name, by_fault in effects.items():
        strata = []
        for values in by_fault.values():
            stacked = np.stack(values)
            count = np.isfinite(stacked).sum(axis=0)
            strata.append(np.divide(np.nansum(stacked, axis=0), count,
                                     out=np.full(count.shape, np.nan), where=count > 0))
        primary[name] = _stratified_block_ci(strata, 24, resamples, 0)
    result = {
        "schema_version": 1, "status": "completed", "dataset": dataset, "scenario": scenario,
        "variant": "hourly_fault_focus", "model_seeds": list(model_seeds), "fault_seeds": list(fault_seeds),
        "bootstrap_unit": "24 selected hourly origins, stratified by fault seed",
        "bootstrap_resamples": resamples,
        "difference_directions": {
            "corel_minus_far_cal": "positive favors FAR-Cal",
            "rolling_minus_corel": "positive favors CoRel",
            "aci_minus_corel": "positive favors CoRel",
        },
        "aggregate_score": {name: float(np.nanmean(np.stack(value))) for name, value in method_scores.items()},
        "aggregate_coverage": {name: float(np.nanmean(np.stack(value))) for name, value in method_coverages.items()},
        "primary_comparisons": primary,
        "far_cal_superior_to_corel": primary["corel_minus_far_cal"]["lower"] > 0,
        "by_run": by_run, "sources": sources,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--dataset", choices=("metr_la", "pems_bay"), required=True)
    parser.add_argument("--scenario", choices=("C3", "C4"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    result = summarize(args.artifacts, args.dataset, args.scenario, args.output, args.resamples)
    print(json.dumps({"dataset": args.dataset, "scenario": args.scenario,
                      "scores": result["aggregate_score"], "coverage": result["aggregate_coverage"],
                      "comparisons": result["primary_comparisons"]}))


if __name__ == "__main__":
    main()
