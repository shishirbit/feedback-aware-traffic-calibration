"""Locked, resumable component ablations on unchanged frozen predictions."""
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ps1.evaluation.farcal_online import evaluate_continuous_farcal, evaluate_sumo_farcal, _stratified_block_ci
from ps1.evaluation.bootstrap import episode_mean_ci
from ps1.evaluation.online_compare import _cache_arrays, _align_origins, _interval_arrays
from ps1.utils.artifacts import sha256_file, write_json
from run_real_core_evaluation_matrix import completed_manifest, write_progress

STAGE = ROOT / "artifacts/ablations/full_registered"

def paths(dataset, scenario, model, fault):
    suffix = "asym_validation-asym100" if dataset == "sumo" and scenario == "C3" else "asym100"
    reference = ROOT / f"artifacts/farcal/{dataset}-seed{model}-{scenario}-fault{fault}-far_cal_asymmetric-{suffix}"
    cache_name = f"asym_validation-{scenario}-fault{fault}" if dataset == "sumo" and scenario == "C3" else f"{scenario}-fault{fault}"
    cache = ROOT / f"artifacts/predictions/{dataset}-seed{model}/{cache_name}"
    fault_dataset = "sumo-asym_validation" if dataset == "sumo" and scenario == "C3" else dataset
    fault_dir = ROOT / f"artifacts/faults/{fault_dataset}/{scenario}-seed{fault}"
    return reference, cache, fault_dir

def output_path(dataset, scenario, model, fault, ablation):
    return STAGE / ablation / f"{dataset}-seed{model}-{scenario}-fault{fault}"

def read_metrics(directory, cache):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    scores, coverage, widths = [], [], []
    for chunk in manifest["chunks"]:
        with np.load(directory / chunk["file"]) as data:
            aligned = _align_origins(cache, data["issue_bin"])
            valid = aligned["original_valid"][:, np.array(manifest["report_horizons"])-1]
            if "evaluation_mask" in data: valid = valid & data["evaluation_mask"][:, None, :]
            metrics = _interval_arrays(aligned["truth_mph"][:, np.array(manifest["report_horizons"])-1],
                data["lower_mph"][:, 0, 0], data["upper_mph"][:, 0, 0], valid, .1)
            if manifest["episode_reset"]:
                scores.append(float(np.nanmean(metrics[0])))
                coverage.append(float(np.nanmean(metrics[1])))
                widths.append(float(np.nanmean(metrics[2])))
            else:
                scores.append(metrics[0]); coverage.append(metrics[1]); widths.append(metrics[2])
    return tuple(np.asarray(x) if manifest["episode_reset"] else np.concatenate(x) for x in (scores, coverage, widths))

def summarize(protocol, runs):
    results = []
    for dataset in protocol["datasets"]:
        for scenario in protocol["scenarios"]:
            paired = {a: [] for a in protocol["ablations"]}
            quality = {a: [] for a in ["full"] + protocol["ablations"]}
            for fault in protocol["fault_seeds"]:
                model_differences = {a: [] for a in protocol["ablations"]}
                for model in protocol["model_seeds"]:
                    reference, cache_path, _ = paths(dataset, scenario, model, fault)
                    cm = json.loads((cache_path / "manifest.json").read_text(encoding="utf-8"))
                    cache = _cache_arrays(cache_path, cm)
                    full = read_metrics(reference, cache)
                    quality["full"].append([float(np.nanmean(x)) for x in full])
                    for ablation in protocol["ablations"]:
                        values = read_metrics(output_path(dataset, scenario, model, fault, ablation), cache)
                        model_differences[ablation].append(values[0]-full[0])
                        quality[ablation].append([float(np.nanmean(x)) for x in values])
                for ablation in protocol["ablations"]:
                    paired[ablation].append(np.nanmean(np.stack(model_differences[ablation]), axis=0))
            comparisons = {}
            for ablation, strata in paired.items():
                ci = episode_mean_ci(np.stack(strata).mean(axis=0)) if dataset == "sumo" else _stratified_block_ci(strata)
                comparisons[ablation] = {"interval_score_ablation_minus_full": ci,
                    "mean_interval_score_coverage_width": np.mean(quality[ablation], axis=0).tolist()}
            results.append({"dataset": dataset, "scenario": scenario, "comparisons": comparisons,
                "full_mean_interval_score_coverage_width": np.mean(quality["full"], axis=0).tolist()})
    write_json(STAGE / "matrix.json", {"status": "completed", "completed": len(runs), "total": protocol["total_stages"],
        "protocol": protocol, "results": results, "runs": runs,
        "inference_label": "Exploratory paired intervals; positive difference means removal worsens interval score."})
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    reports = ROOT / "reports"
    figure_dir = reports / "figures/publication"
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = {"no_spatial": "No spatial pooling", "no_context": "No context matching", "no_age": "No target-age weights",
        "no_stale_blend": "No stale blend", "no_inflation": "No inflation", "arrival_freshness": "Arrival-time freshness", "frozen_only": "Frozen archive only"}
    rows = []; lines = ["# Full registered component ablations", "", "All 432 stages completed on unchanged FAR-GW prediction caches.", "",
        "Positive paired interval-score differences mean removing the component worsened performance. Intervals are exploratory 95% bootstrap intervals; overlapping or negative intervals are reported without a superiority claim.", "",
        "Real-data inference uses day blocks within fault seeds after averaging model seeds. SUMO inference resamples episodes after averaging model and fault seeds. C3 uses all 40 fresh validation episodes; C4 uses all 30 test episodes.", "",
        "| Dataset | Scenario | Ablation | Score difference (95% CI) | Coverage | Width (mph) |",
        "|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for result in results:
        column = protocol["datasets"].index(result["dataset"]); row = protocol["scenarios"].index(result["scenario"])
        ax = axes[row, column]
        for index, ablation in enumerate(protocol["ablations"]):
            comparison = result["comparisons"][ablation]; ci = comparison["interval_score_ablation_minus_full"]
            score, coverage, width = comparison["mean_interval_score_coverage_width"]
            rows.append({"dataset": result["dataset"], "scenario": result["scenario"], "ablation": ablation,
                "difference": ci["estimate"], "lower": ci["lower"], "upper": ci["upper"], "interval_score": score, "coverage": coverage, "width_mph": width})
            lines.append(f"| {result['dataset']} | {result['scenario']} | {labels[ablation]} | {ci['estimate']:.3f} [{ci['lower']:.3f}, {ci['upper']:.3f}] | {coverage:.3f} | {width:.3f} |")
            ax.plot([ci["lower"], ci["upper"]], [index, index], color="#356D95", linewidth=1.5)
            ax.plot(ci["estimate"], index, "o", color="#356D95", markersize=4)
        ax.axvline(0, color="0.4", linewidth=.8, linestyle="--")
        ax.set_title(f"{result['dataset']} · {result['scenario']}")
        ax.set_yticks(range(7), [labels[a] for a in protocol["ablations"]])
        ax.set_xlabel("Ablation − full interval score (mph)")
        ax.grid(axis="x", alpha=.2)
    axes[0, 0].invert_yaxis()
    fig.tight_layout()
    prefix = figure_dir / "fig15-full-component-ablations"
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    with prefix.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    lines.extend(["", "The arrival-time diagnostic changes freshness weighting and the stale gap only; buffer eligibility and ESS blocks remain target-based. The frozen-only comparator removes live adaptation and adaptive inflation.", "",
        "These results concern FAR-GW component ablations. The separately completed four-backbone main comparison includes STAEformer. No universal superiority or identifiable MNAR correction is claimed."])
    (reports / "full-registered-ablations.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    protocol_path = ROOT / "configs/full_registered_ablations.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source_path = ROOT / "src/ps1/evaluation/farcal_online.py"
    provenance_path = STAGE / "protocol.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance["protocol_sha256"] != sha256_file(protocol_path) or provenance["evaluator_sha256"] != sha256_file(source_path):
            raise ValueError("locked ablation protocol or evaluator changed")
    else:
        write_json(provenance_path, {"protocol": protocol, "protocol_sha256": sha256_file(protocol_path),
            "evaluator_sha256": sha256_file(source_path), "runner_sha256": sha256_file(Path(__file__))})
    runs = []; started = time.time()
    try:
        for dataset in protocol["datasets"]:
            for scenario in protocol["scenarios"]:
                for model in protocol["model_seeds"]:
                    for fault in protocol["fault_seeds"]:
                        reference, cache, fault_dir = paths(dataset, scenario, model, fault)
                        if not completed_manifest(reference): raise ValueError(f"invalid reference {reference}")
                        rm = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
                        calibration = ROOT / f"artifacts/predictions/{dataset}-seed{model}/calibration"
                        for key, source in (("calibration_manifest_sha256", calibration / "manifest.json"),
                                            ("test_manifest_sha256", cache / "manifest.json"),
                                            ("fault_manifest_sha256", fault_dir / "fault_manifest.json")):
                            if sha256_file(source) != rm[key]: raise ValueError(f"reference input changed: {source}")
                        for ablation in ["full"] + protocol["ablations"]:
                            output = reference if ablation == "full" else output_path(dataset, scenario, model, fault, ablation)
                            name = f"{dataset}/{scenario}/model{model}/fault{fault}/{ablation}"
                            write_progress(STAGE / "progress.json", {"status": "running", "completed": len(runs),
                                "total": protocol["total_stages"], "current": name, "updated_unix": time.time(), "runs": runs})
                            existed = completed_manifest(output)
                            if not existed:
                                config = {"candidate_id": "asym100", "parameters": {**rm["parameters"], "ablation": ablation},
                                    "prepared_path": str(ROOT / f"data/prepared/{dataset}.npz")}
                                for key in ("evaluation_stride", "focus_fault_events", "focus_recovery_bins"):
                                    if key in rm: config[key] = rm[key]
                                evaluator = evaluate_sumo_farcal if dataset == "sumo" else evaluate_continuous_farcal
                                evaluator(calibration, cache, fault_dir,
                                    output, config, methods=("far_cal_asymmetric",))
                            runs.append({"run": name, "status": "verified_existing" if existed else "completed",
                                "manifest": str(output / "manifest.json"), "sha256": sha256_file(output / "manifest.json")})
                            print(f"[{len(runs)}/{protocol['total_stages']}] {name}", flush=True)
        summarize(protocol, runs)
        write_progress(STAGE / "progress.json", {"status": "completed", "completed": len(runs),
            "total": protocol["total_stages"], "elapsed_seconds": time.time()-started, "updated_unix": time.time(), "runs": runs})
    except BaseException as error:
        write_progress(STAGE / "progress.json", {"status": "failed", "completed": len(runs), "total": protocol["total_stages"],
            "error": repr(error), "updated_unix": time.time(), "runs": runs})
        raise

if __name__ == "__main__": main()
