from pathlib import Path
import hashlib
import json
import yaml


VALID_STATUSES={"planned","running","completed","failed","blocked"}


def load_registry(path,root):
    path=Path(path); root=Path(root)
    registry=yaml.safe_load(path.read_text(encoding="utf-8"))
    if registry.get("schema_version")!=1 or not isinstance(registry.get("experiments"),list):
        raise ValueError("unsupported experiment registry")
    ids=set()
    for experiment in registry["experiments"]:
        if experiment.get("id") in ids: raise ValueError("duplicate experiment ID")
        ids.add(experiment.get("id"))
        if experiment.get("status") not in VALID_STATUSES: raise ValueError("invalid experiment status")
        if experiment["status"]=="completed":
            artifact=experiment.get("artifact_path")
            if not artifact or not (root/artifact).exists():
                raise FileNotFoundError(f"completed experiment lacks artifact: {experiment.get('id')}")
        elif not experiment.get("reason") and not experiment.get("note"):
            raise ValueError(f"non-completed experiment needs reason or note: {experiment.get('id')}")
    return registry


def render_status_report(registry_path,root,output_dir):
    registry_path,root,output_dir=Path(registry_path),Path(root),Path(output_dir)
    registry=load_registry(registry_path,root)
    digest=hashlib.sha256(registry_path.read_bytes()).hexdigest()
    lines=["# PS-1 evidence status","",f"Registry SHA-256: `{digest}`","",
           "This report distinguishes executed software checks from pending research evidence.","",
           "| Experiment | Dataset | Status | Evidence |","|---|---|---|---|"]
    for experiment in registry["experiments"]:
        evidence=experiment.get("artifact_path") or experiment.get("reason") or experiment.get("note")
        lines.append(f"| {experiment['id']} | {experiment.get('dataset','')} | {experiment['status']} | {evidence} |")
    lines.extend(["","## Scientific claim status","",
                  "Scientific claims are bounded by the completed registry entries and their cited artifacts. Planned or blocked entries remain outside the supported claim set; the current synthesis is maintained in reports/empirical-summary.md and reports/publication-readiness.md.","",
                  "No missing result has been replaced by a generated number or zero.",""])
    output_dir.mkdir(parents=True,exist_ok=True)
    output=output_dir/f"status-{digest[:12]}.md"; payload="\n".join(lines)
    if output.exists() and output.read_text(encoding="utf-8")!=payload:
        raise FileExistsError(f"refusing to overwrite differing report: {output}")
    output.write_text(payload,encoding="utf-8")
    return output


def render_clean_report(evaluation_path,output_dir):
    evaluation_path,output_dir=Path(evaluation_path),Path(output_dir)
    result=json.loads(evaluation_path.read_text(encoding="utf-8"))
    if result.get("status")!="completed_clean_C0_only": raise ValueError("not a completed clean evaluation")
    digest=hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
    lines=[f"# {result['dataset'].upper()} clean C0 evidence — seed {result['training_seed']}","",
           f"Evaluation SHA-256: `{digest}`","",
           "This is single-seed clean-input/clean-feedback evidence. It does not evaluate H1-H4.","",
           "| Horizon | Targets | MAE (mph) | RMSE (mph) | Raw 90% PICP | Raw MPIW | Frozen 90% PICP | Frozen MPIW | Frozen interval score |","|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for horizon in result["report_horizons"]:
        key=str(horizon); point=result["point"][key]; raw=result["intervals"]["raw_q05_q95"][key]; frozen=result["intervals"]["frozen"]["0.9"][key]
        lines.append(f"| {horizon} | {point['count']:,} | {point['mae_mph']:.4f} | {point['rmse_mph']:.4f} | {raw['picp']:.4f} | {raw['mpiw_mph']:.4f} | {frozen['picp']:.4f} | {frozen['mpiw_mph']:.4f} | {frozen['mean_interval_score']:.4f} |")
    lines.extend(["",f"Equal-horizon mean frozen 90% interval score: **{result['intervals']['primary_equal_horizon_mean_frozen_90_interval_score']:.4f}**.","",
                  "Limitations: "+"; ".join(result["limitations"])+".",""])
    output_dir.mkdir(parents=True,exist_ok=True); output=output_dir/f"clean-{result['dataset']}-seed{result['training_seed']}-{digest[:12]}.md"
    payload="\n".join(lines)
    if output.exists() and output.read_text(encoding="utf-8")!=payload: raise FileExistsError("refusing to overwrite differing clean report")
    output.write_text(payload,encoding="utf-8"); return output


def render_feedback_report(evaluation_path,output_dir):
    evaluation_path,output_dir=Path(evaluation_path),Path(output_dir); result=json.loads(evaluation_path.read_text(encoding="utf-8"))
    if result.get("status")!="completed_single_seed_single_fault": raise ValueError("not a completed paired feedback comparison")
    digest=hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
    interval_label="episode-bootstrap" if result.get("bootstrap_unit")=="episode" else "block-bootstrap"
    lines=[f"# {result['dataset'].upper()} paired feedback timing — model seed {result['model_seed']}, fault seed {result['fault_seed']}","",
           f"Evaluation SHA-256: `{digest}`","",f"Point/input scenario: {result['point_scenario']}. Feedback comparison: {result['immediate_feedback_scenario']} (immediate) versus {result['delayed_feedback_scenario']} (delayed).","",
           "Positive differences mean delayed feedback has a worse interval score.","",
           f"| Method | Equal-horizon 90% score difference | 95% {interval_label} interval |","|---|---:|---:|"]
    for method,comparison in result["comparisons"].items():
        primary=comparison["primary_equal_horizon_90_interval_score_delayed_minus_immediate"]
        lines.append(f"| {method} | {primary['estimate']:.7f} | [{primary['lower']:.7f}, {primary['upper']:.7f}] |")
    lines.extend(["","This is preliminary H1 evidence only. "+"; ".join(result["limitations"])+".",""])
    output_dir.mkdir(parents=True,exist_ok=True); output=output_dir/f"feedback-{result['dataset']}-seed{result['model_seed']}-fault{result['fault_seed']}-{digest[:12]}.md"
    payload="\n".join(lines)
    if output.exists() and output.read_text(encoding="utf-8")!=payload: raise FileExistsError("refusing to overwrite differing feedback report")
    output.write_text(payload,encoding="utf-8"); return output
