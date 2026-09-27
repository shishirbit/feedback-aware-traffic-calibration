import argparse
from pathlib import Path
import json
import numpy as np
from ps1.config import resolve, config_hash
from ps1.simulation.network import build_network
from ps1.simulation.demand import generate_routes
from ps1.simulation.runner import write_config, run_offline, parse_e1
from ps1.utils.artifacts import environment_manifest, write_json
from ps1.smoke import train_smoke, smoke_calibration_eval
from ps1.data.provenance import audit_hdf
from ps1.data.prepare_real import prepare_hdf
from ps1.data.prepare_sumo import prepare_sumo_paper,prepare_sumo_replication
from ps1.online.faults import generate_fault_schedule
from ps1.simulation.live import run_live
from ps1.simulation.paper import generate_paper_dataset,generate_replication_dataset
from ps1.simulation.sumo_faults import generate_sumo_faults,derive_immediate_feedback_oracle
from ps1.evaluation.report import render_status_report,render_clean_report,render_feedback_report
from ps1.training import train_registered, feasibility_batch
from ps1.prediction_cache import cache_clean_predictions,cache_faulted_predictions,compare_prediction_caches
from ps1.evaluation.clean_baselines import evaluate_clean_baselines,evaluate_simple_controls
from ps1.evaluation.online_compare import evaluate_received_online,compare_feedback_arms,compare_input_mechanism
from ps1.evaluation.recovery import evaluate_sumo_recovery,summarize_sumo_recovery
from ps1.evaluation.farcal_online import evaluate_sumo_farcal,evaluate_continuous_farcal,summarize_sumo_farcal_ablation,summarize_locked_sumo_farcal,summarize_sumo_replication,summarize_real_transfer

ROOT=Path(__file__).resolve().parents[2]


def load(profile, dataset=None):
    paths=[]
    if profile:
        paths.append(ROOT / profile)
    if dataset:
        paths.append(ROOT / "configs" / f"{dataset}.yaml")
    return resolve(ROOT / "configs" / "base.yaml", *paths)


def cmd_audit_environment(args):
    print(json.dumps(environment_manifest(),indent=2))


def cmd_audit_data(args):
    config=load(None,args.dataset)
    data=config["data"]
    audit=audit_hdf(ROOT/data["filename"],data["expected_sensors"],data["source_url"],ROOT/data["adjacency_filename"],
                    retrieval_url=data["retrieval_url"],license_access_terms=data["license_access_terms"])
    output=ROOT/"artifacts"/"data-audit"/f"{args.dataset}-v3.json"
    write_json(output,audit)
    print(json.dumps(audit,indent=2))


def cmd_prepare(args):
    config=load(None,args.dataset); data=config["data"]
    output=ROOT/"data"/"prepared"/f"{args.dataset}.npz"
    manifest=prepare_hdf(ROOT/data["filename"],ROOT/data["adjacency_filename"],output,data["expected_sensors"])
    print(json.dumps(manifest,indent=2))


def cmd_prepare_sumo(args):
    config=load(None,"sumo")
    warmup_bins=config["sumo"]["warmup_seconds"]//config["sumo"]["aggregation_seconds"]
    manifest=prepare_sumo_paper(ROOT/args.input,ROOT/args.output,warmup_bins)
    print(json.dumps({key:manifest[key] for key in ("artifact_sha256","rows","episodes","split_episode_counts","family_episode_counts")},indent=2))


def cmd_prepare_sumo_replication(args):
    manifest=prepare_sumo_replication(ROOT/args.input,ROOT/args.reference,ROOT/args.output)
    print(json.dumps({key:manifest[key] for key in ("status","artifact_sha256","rows","episodes","replication_episode_indices")},indent=2))


def cmd_make_sumo_faults(args):
    config=load(None,"sumo"); prepared=ROOT/(f"data/prepared/sumo_{args.partition}.npz" if args.partition in {"replication","confirmation","asym_validation"} else "data/prepared/sumo.npz")
    fault_root="sumo" if args.partition=="test" else f"sumo-{args.partition}"
    output=ROOT/"artifacts/faults"/fault_root/f"{args.scenario}-seed{args.seed}"
    result=generate_sumo_faults(prepared,prepared.with_suffix(".manifest.json"),args.scenario,args.seed,output,config,args.partition)
    print(json.dumps({key:result[key] for key in ("scenario_id","seed","event_count","affected_episode_indices","realized")},indent=2))


def cmd_make_sumo_feedback_oracle(args):
    base=ROOT/"artifacts/faults/sumo"
    result=derive_immediate_feedback_oracle(base/f"C4-seed{args.seed}",base/f"C4I-seed{args.seed}")
    print(json.dumps({key:result[key] for key in ("scenario_id","seed","oracle_role","realized")},indent=2))


def _write_faults(dataset,scenario,seed):
    prepared=ROOT/"data"/"prepared"/f"{dataset}.npz"
    with np.load(prepared) as data:
        timestamps=data["timestamp_ns"]; adjacency=data["adjacency"]
    config=load(None,dataset); boundaries=json.loads((prepared.with_suffix(".manifest.json")).read_text(encoding="utf-8"))["boundary_timestamps"]
    test_start=np.datetime64(boundaries[3]).astype("datetime64[ns]").astype("int64")
    start=int(np.searchsorted(timestamps,test_start)); length=len(timestamps)-start
    faults=config["faults"]
    schedule=generate_fault_schedule(scenario,length,adjacency,seed,faults["connected_fraction"],faults["duration_bins"],faults["backlog_release_bins"],faults["events_per_day"])
    output=ROOT/"artifacts"/"faults"/dataset/f"{scenario}-seed{seed}"
    output.mkdir(parents=True,exist_ok=True)
    schedule_path=output/"release_schedule.npz"
    if schedule_path.exists():
        with np.load(schedule_path) as existing:
            if not (np.array_equal(existing["input_arrival"],schedule.input_arrival) and np.array_equal(existing["feedback_arrival"],schedule.feedback_arrival)):
                raise FileExistsError(f"refusing to overwrite differing fault schedule: {schedule_path}")
    else:
        np.savez_compressed(schedule_path,input_arrival=schedule.input_arrival,feedback_arrival=schedule.feedback_arrival)
    manifest=schedule.manifest(); manifest.update({"dataset":dataset,"test_start_row":start,"test_length":length})
    write_json(output/"fault_manifest.json",manifest)
    return {"scenario_id":scenario,"dataset":dataset,"seed":seed,
                      "event_count":len(schedule.events),"realized":manifest["realized"],
                      "artifact_path":str(output)}


def cmd_make_faults(args):
    print(json.dumps(_write_faults(args.dataset,args.scenario,args.seed),indent=2))


def cmd_make_registered_faults(args):
    results=[]
    for dataset in ("metr_la","pems_bay"):
        for scenario in (f"C{i}" for i in range(6)):
            for seed in (101,202,303):
                results.append(_write_faults(dataset,scenario,seed))
    print(json.dumps({"generated":len(results),"datasets":["metr_la","pems_bay"],"scenarios":"C0-C5","seeds":[101,202,303]},indent=2))


def cmd_sumo_live(args):
    config=load(args.config); network=ROOT/"data/sumo/network"; episode=ROOT/"data/sumo/live-smoke"
    build_network(network,config["sumo"]["mainline_length_m"],20,2)
    generate_routes(network,episode,args.seconds,args.seed,args.family)
    sumocfg=write_config(network,episode,args.seconds,args.seed)
    events=json.loads((episode/"episode_manifest.json").read_text(encoding="utf-8"))["physical_events"]
    result=run_live(sumocfg,episode,20,2,events)
    write_json(episode/"live_equivalence.json",result); print(json.dumps(result,indent=2))


def cmd_report(args):
    output=render_status_report(ROOT/args.registry,ROOT,ROOT/"reports")
    print(str(output))


def cmd_train(args):
    config=load(None,args.dataset)
    if args.seed not in config["training"]["seeds"]:
        raise ValueError("seed is not in the registered training seed set")
    prepared=ROOT/"data"/"prepared"/f"{args.dataset}.npz"
    output=ROOT/"artifacts"/"training"/f"{args.dataset}-seed{args.seed}-{config_hash(config)[:12]}"
    print(json.dumps(train_registered(prepared,output,config,args.seed),indent=2))


def cmd_train_feasibility(args):
    config=load(None,args.dataset)
    prepared=ROOT/"data"/"prepared"/f"{args.dataset}.npz"
    result=feasibility_batch(prepared,config,args.seed)
    output=ROOT/"artifacts"/"feasibility"/f"{args.dataset}-seed{args.seed}.json"
    write_json(output,result); print(json.dumps(result,indent=2))


def cmd_cache_predictions(args):
    config=load(None,args.dataset)
    if args.seed not in config["training"]["seeds"]:
        raise ValueError("seed is not in the registered training seed set")
    prepared=ROOT/"data"/"prepared"/f"{args.dataset}.npz"
    training=ROOT/"artifacts"/"training"/f"{args.dataset}-seed{args.seed}-{config_hash(config)[:12]}"
    checkpoint=training/"checkpoint.pt"
    output=ROOT/"artifacts"/"predictions"/f"{args.dataset}-seed{args.seed}"/args.partition
    result=cache_clean_predictions(prepared,checkpoint,output,args.partition,config["training"]["batch_size"])
    print(json.dumps(result,indent=2))


def cmd_cache_faulted(args):
    config=load(None,args.dataset)
    if args.model_seed not in config["training"]["seeds"] or args.fault_seed not in config["faults"]["seeds"]:
        raise ValueError("model or fault seed is not registered")
    prepared=ROOT/"data"/"prepared"/(f"sumo_{args.partition}.npz" if args.dataset=="sumo" and args.partition in {"replication","confirmation","asym_validation"} else f"{args.dataset}.npz")
    checkpoint=ROOT/"artifacts"/"training"/f"{args.dataset}-seed{args.model_seed}-{config_hash(config)[:12]}"/"checkpoint.pt"
    fault_root=args.dataset if args.partition=="test" else f"{args.dataset}-{args.partition}"
    fault_dir=ROOT/"artifacts"/"faults"/fault_root/f"{args.scenario}-seed{args.fault_seed}"
    output_name=f"{args.scenario}-fault{args.fault_seed}" if args.partition=="test" else f"{args.partition}-{args.scenario}-fault{args.fault_seed}"
    output=ROOT/"artifacts"/"predictions"/f"{args.dataset}-seed{args.model_seed}"/output_name
    print(json.dumps(cache_faulted_predictions(prepared,checkpoint,fault_dir,output,config["training"]["batch_size"],partition=args.partition),indent=2))


def cmd_validate_c0_cache(args):
    base=ROOT/"artifacts"/"predictions"/f"{args.dataset}-seed{args.model_seed}"
    candidate=base/f"C0-fault{args.fault_seed}"
    output=ROOT/"artifacts"/"validation"/f"{args.dataset}-seed{args.model_seed}-C0-fault{args.fault_seed}.json"
    print(json.dumps(compare_prediction_caches(base/"test",candidate,output),indent=2))


def cmd_evaluate_clean(args):
    base=ROOT/"artifacts"/"predictions"/f"{args.dataset}-seed{args.seed}"
    output=ROOT/"artifacts"/"evaluation"/f"{args.dataset}-seed{args.seed}-clean.json"
    result=evaluate_clean_baselines(base/"calibration",base/"test",output)
    print(json.dumps(result,indent=2))


def cmd_evaluate_static_fault(args):
    base=ROOT/"artifacts"/"predictions"/f"{args.dataset}-seed{args.model_seed}"
    test=base/f"{args.scenario}-fault{args.fault_seed}"
    output=ROOT/"artifacts"/"evaluation"/f"{args.dataset}-seed{args.model_seed}-{args.scenario}-fault{args.fault_seed}-static.json"
    print(json.dumps(evaluate_clean_baselines(base/"calibration",test,output),indent=2))


def cmd_evaluate_online(args):
    base=ROOT/"artifacts"/"predictions"/f"{args.dataset}-seed{args.model_seed}"
    if args.partition=="tune" and args.dataset!="sumo": raise ValueError("split-safe tuning replay is currently SUMO-only")
    fault_root=args.dataset if args.partition=="test" else f"sumo-{args.partition}"
    point_fault=ROOT/"artifacts"/"faults"/fault_root/f"{args.point_scenario}-seed{args.fault_seed}"
    feedback_fault=ROOT/"artifacts"/"faults"/fault_root/f"{args.feedback_scenario}-seed{args.fault_seed}"
    subset=None; calibration_subset=None; suffix=""
    if args.episode_subset:
        subset_spec=json.loads((ROOT/args.episode_subset).read_text(encoding="utf-8"))
        subset=subset_spec["episode_indices"]; suffix=f"-{subset_spec['artifact_suffix']}"
    cache_name=f"{args.point_scenario}-fault{args.fault_seed}"
    calibration_cache=base/"calibration"
    if args.partition=="tune":
        if subset is not None: raise ValueError("tuning uses the registered split")
        episodes=json.loads((point_fault/"fault_manifest.json").read_text(encoding="utf-8"))["test_episode_indices"]
        cut=len(episodes)//3; calibration_subset=episodes[:cut]; subset=episodes[cut:]; suffix="-tune-splitsafe"
        cache_name=f"tune-{cache_name}"; calibration_cache=base/cache_name
    elif args.partition in {"replication","confirmation","asym_validation"}:
        cache_name=f"{args.partition}-{cache_name}"; suffix=f"-{args.partition}"
    output=ROOT/"artifacts"/"online"/f"{args.dataset}-seed{args.model_seed}-{args.point_scenario}-feedback-{args.feedback_scenario}-fault{args.fault_seed}{suffix}"
    result=evaluate_received_online(calibration_cache,base/cache_name,point_fault,feedback_fault,output,episode_subset=subset,
        calibration_episode_subset=calibration_subset)
    print(json.dumps(result,indent=2))


def cmd_compare_feedback(args):
    base=ROOT/"artifacts"/"online"
    suffix=""
    if args.episode_subset:
        subset_spec=json.loads((ROOT/args.episode_subset).read_text(encoding="utf-8")); suffix=f"-{subset_spec['artifact_suffix']}"
    immediate=base/f"{args.dataset}-seed{args.model_seed}-{args.point_scenario}-feedback-{args.immediate_scenario}-fault{args.fault_seed}{suffix}"
    delayed=base/f"{args.dataset}-seed{args.model_seed}-{args.point_scenario}-feedback-{args.delayed_scenario}-fault{args.fault_seed}{suffix}"
    test=ROOT/"artifacts"/"predictions"/f"{args.dataset}-seed{args.model_seed}"/f"{args.point_scenario}-fault{args.fault_seed}"
    output=ROOT/"artifacts"/"evaluation"/f"{args.dataset}-seed{args.model_seed}-{args.point_scenario}-feedback-{args.immediate_scenario}-vs-{args.delayed_scenario}-fault{args.fault_seed}{suffix}.json"
    config=load(None,args.dataset)
    print(json.dumps(compare_feedback_arms(immediate,delayed,test,output,config["evaluation"]["bootstrap_resamples"],config["evaluation"]["real_block_hours"]*12),indent=2))


def cmd_report_feedback(args):
    source=ROOT/"artifacts"/"evaluation"/f"{args.dataset}-seed{args.model_seed}-{args.point_scenario}-feedback-{args.immediate_scenario}-vs-{args.delayed_scenario}-fault{args.fault_seed}.json"
    print(render_feedback_report(source,ROOT/"reports"))


def cmd_compare_sumo_input(args):
    subset=json.loads((ROOT/args.episode_subset).read_text(encoding="utf-8")); suffix=subset["artifact_suffix"]
    online=ROOT/"artifacts"/"online"; predictions=ROOT/"artifacts"/"predictions"/f"sumo-seed{args.model_seed}"
    reference=online/f"sumo-seed{args.model_seed}-C0-feedback-C0-fault{args.fault_seed}-{suffix}"
    candidate=online/f"sumo-seed{args.model_seed}-C2-feedback-C2-fault{args.fault_seed}-{suffix}"
    output=ROOT/"artifacts"/"evaluation"/f"sumo-seed{args.model_seed}-C2-input-mechanism-fault{args.fault_seed}-{suffix}.json"
    print(json.dumps(compare_input_mechanism(reference,candidate,predictions/f"C0-fault{args.fault_seed}",predictions/f"C2-fault{args.fault_seed}",output),indent=2))


def cmd_evaluate_sumo_recovery(args):
    config=load(None,"sumo"); online=ROOT/"artifacts/online"
    oracle="C2" if args.scenario=="C3" else "C4I"
    candidate=online/f"sumo-seed{args.model_seed}-{args.scenario}-feedback-{args.scenario}-fault{args.fault_seed}"
    oracle_dir=online/f"sumo-seed{args.model_seed}-{args.scenario}-feedback-{oracle}-fault{args.fault_seed}"
    test=ROOT/"artifacts/predictions"/f"sumo-seed{args.model_seed}"/f"{args.scenario}-fault{args.fault_seed}"
    fault=ROOT/"artifacts/faults/sumo"/f"{args.scenario}-seed{args.fault_seed}"
    output=ROOT/"artifacts/evaluation"/f"sumo-seed{args.model_seed}-{args.scenario}-recovery-fault{args.fault_seed}.json"
    settings=config["evaluation"]
    result=evaluate_sumo_recovery(candidate,oracle_dir,test,fault,output,settings["recovery_window_origins"],
        settings["recovery_min_pairs"],settings["recovery_tolerance"],settings["recovery_consecutive_windows"])
    print(json.dumps({key:result[key] for key in ("scenario","model_seed","fault_seed","record_count","coverage_censored_count","joint_censored_count")},indent=2))


def cmd_summarize_sumo_recovery(args):
    result=summarize_sumo_recovery(ROOT/"artifacts/evaluation",ROOT/"artifacts/evaluation/sumo-recovery-summary.json")
    print(json.dumps({key:result[key] for key in ("runs","record_count","coverage_censored_count","joint_censored_count","maximum_window_pairs","registered_minimum_pairs")},indent=2))


def cmd_evaluate_sumo_farcal(args):
    config=load(None,"sumo"); base=ROOT/"artifacts/predictions"/f"sumo-seed{args.model_seed}"
    settings=config["calibration"]
    evaluator={"prepared_path":ROOT/"data/prepared/sumo.npz","parameters":{
        "tau":settings["age_decay_bins"],"bandwidth":settings["context_bandwidth"],"support":settings["support_reference"],
        "stale_decay":settings["stale_decay_bins"],"beta_gap":settings["beta_gap"],"beta_missing":settings["beta_missing"],
        "inflation_cap":settings["inflation_cap"],"buffer_bins":settings["buffer_days"]*288}}
    candidate_suffix=""
    if args.tuning_candidate:
        registries=("sumo_farcal_tuning_candidates.json","sumo_farcal_v2_candidates.json","sumo_farcal_asymmetric_candidates.json")
        candidates=[row for name in registries if (ROOT/"configs"/name).exists() for row in json.loads((ROOT/"configs"/name).read_text(encoding="utf-8"))["candidates"]]
        selected=next((row for row in candidates if row["id"]==args.tuning_candidate),None)
        if selected is None: raise ValueError("unknown FAR-Cal tuning candidate")
        evaluator["parameters"].update(selected["parameters"]); evaluator["candidate_id"]=selected["id"]; candidate_suffix=f"-{selected['id']}"
    fault_root="sumo" if args.partition=="test" else f"sumo-{args.partition}"
    fault=ROOT/"artifacts/faults"/fault_root/f"{args.scenario}-seed{args.fault_seed}"
    subset=None; calibration_subset=None; suffix=""
    if args.episode_subset:
        subset_spec=json.loads((ROOT/args.episode_subset).read_text(encoding="utf-8")); subset=subset_spec["episode_indices"]; suffix=f"-{subset_spec['artifact_suffix']}"
    if args.partition=="tune":
        if subset is not None: raise ValueError("tuning uses the registered first-third warm-up and remaining-two-thirds scoring split")
        tune_episodes=json.loads((fault/"fault_manifest.json").read_text(encoding="utf-8"))["test_episode_indices"]
        warmup_count=len(tune_episodes)//3; calibration_subset=tune_episodes[:warmup_count]; subset=tune_episodes[warmup_count:]
        suffix="-splitsafe"
    methods=tuple(args.methods.split(",")); method_suffix="allmethods" if methods==tuple(("far_cal","no_spatial","no_context","no_age","no_stale_blend","no_inflation")) else "-".join(methods)
    partition_suffix="" if args.partition=="test" else f"-{args.partition}"
    cache_name=f"{args.scenario}-fault{args.fault_seed}" if args.partition=="test" else f"{args.partition}-{args.scenario}-fault{args.fault_seed}"
    output=ROOT/"artifacts/farcal"/f"sumo-seed{args.model_seed}-{args.scenario}-fault{args.fault_seed}-{method_suffix}{partition_suffix}{suffix}{candidate_suffix}"
    calibration_cache=base/cache_name if args.partition=="tune" else base/"calibration"
    result=evaluate_sumo_farcal(calibration_cache,base/cache_name,fault,output,evaluator,episode_subset=subset,methods=methods,
        calibration_episode_subset=calibration_subset)
    print(json.dumps({"status":result["status"],"model_seed":result["model_seed"],"scenario":result["scenario"],"fault_seed":result["fault_seed"],"episode_count":result["episode_count"]},indent=2))


def cmd_evaluate_real_farcal(args):
    config=load(None,args.dataset); settings=config["calibration"]
    locked=json.loads((ROOT/"configs/sumo_farcal_asymmetric_locked.json").read_text(encoding="utf-8"))
    parameters={"tau":settings["age_decay_bins"],"bandwidth":settings["context_bandwidth"],"support":settings["support_reference"],
        "stale_decay":settings["stale_decay_bins"],"beta_gap":settings["beta_gap"],"beta_missing":settings["beta_missing"],
        "inflation_cap":settings["inflation_cap"],"buffer_bins":settings["buffer_days"]*288,**locked["parameters"]}
    parameters.update({"scalable_local_only":True,"records_per_sensor_cap":256})
    evaluator={"prepared_path":ROOT/"data/prepared"/f"{args.dataset}.npz","parameters":parameters,"candidate_id":locked["selected_candidate"],"evaluation_stride":12,
        "focus_fault_events":True,"focus_recovery_bins":12}
    base=ROOT/"artifacts/predictions"/f"{args.dataset}-seed{args.model_seed}"
    fault=ROOT/"artifacts/faults"/args.dataset/f"{args.scenario}-seed{args.fault_seed}"
    output=ROOT/"artifacts/farcal"/f"{args.dataset}-seed{args.model_seed}-{args.scenario}-fault{args.fault_seed}-far_cal_asymmetric-asym100"
    result=evaluate_continuous_farcal(base/"calibration",base/f"{args.scenario}-fault{args.fault_seed}",fault,output,evaluator)
    print(json.dumps({"status":result["status"],"dataset":result["dataset"],"model_seed":result["model_seed"],"scenario":result["scenario"],"fault_seed":result["fault_seed"]},indent=2))


def cmd_summarize_real_transfer(args):
    model_seeds=(11,22,33)
    result=summarize_real_transfer(ROOT/"artifacts/farcal",ROOT/"artifacts/online",ROOT/"artifacts/predictions",
        ROOT/"artifacts/evaluation"/f"{args.dataset}-asymmetric-{args.scenario}-transfer-audit.json",args.dataset,model_seeds=model_seeds,scenario=args.scenario)
    print(json.dumps({key:result[key] for key in ("status","dataset","far_cal_interval_score","far_cal_coverage","primary_comparisons","superiority_rule_met")},indent=2))


def cmd_summarize_sumo_farcal(args):
    result=summarize_sumo_farcal_ablation(ROOT/"artifacts/farcal",ROOT/"artifacts/online",ROOT/"artifacts/predictions",ROOT/"artifacts/evaluation/sumo-C3-farcal-ablation-subset12.json")
    print(json.dumps({"status":result["status"],"model_seeds":result["model_seeds"],"between_seed_mean_effect":result["between_seed_mean_effect"]},indent=2))


def cmd_summarize_locked_sumo_farcal(args):
    result=summarize_locked_sumo_farcal(ROOT/"artifacts/farcal",ROOT/"artifacts/online",ROOT/"artifacts/predictions",
        ROOT/"artifacts/evaluation/sumo-C3-farcal-locked-full.json",args.candidate)
    print(json.dumps({"status":result["status"],"candidate_id":result["candidate_id"],
        "mean_effect_across_runs":result["mean_effect_across_runs"],"runs_with_ci_above_zero":result["runs_with_ci_above_zero"]},indent=2))


def cmd_summarize_sumo_replication(args):
    result=summarize_sumo_replication(ROOT/"artifacts/farcal",ROOT/"artifacts/online",ROOT/"artifacts/predictions",
        ROOT/"artifacts/evaluation/sumo-independent-replication.json",args.candidate)
    print(json.dumps({key:result[key] for key in ("status","candidate_id","far_cal_interval_score","far_cal_coverage","primary_comparisons","superiority_rule_met")},indent=2))


def cmd_summarize_sumo_confirmation(args):
    result=summarize_sumo_replication(ROOT/"artifacts/farcal",ROOT/"artifacts/online",ROOT/"artifacts/predictions",
        ROOT/"artifacts/evaluation/sumo-powered-confirmation.json",args.candidate,partition="confirmation")
    print(json.dumps({key:result[key] for key in ("status","candidate_id","far_cal_interval_score","far_cal_coverage","primary_comparisons","superiority_rule_met")},indent=2))


def cmd_summarize_asymmetric_development(args):
    result=summarize_sumo_replication(ROOT/"artifacts/farcal",ROOT/"artifacts/online",ROOT/"artifacts/predictions",
        ROOT/"artifacts/evaluation/sumo-asymmetric-development.json",args.candidate,method="far_cal_asymmetric")
    print(json.dumps({key:result[key] for key in ("status","candidate_id","far_cal_interval_score","far_cal_coverage","primary_comparisons","superiority_rule_met")},indent=2))


def cmd_summarize_asymmetric_validation(args):
    result=summarize_sumo_replication(ROOT/"artifacts/farcal",ROOT/"artifacts/online",ROOT/"artifacts/predictions",
        ROOT/"artifacts/evaluation/sumo-asymmetric-validation.json",args.candidate,partition="asym_validation",method="far_cal_asymmetric")
    print(json.dumps({key:result[key] for key in ("status","candidate_id","far_cal_interval_score","far_cal_coverage","primary_comparisons","superiority_rule_met")},indent=2))


def cmd_summarize_asymmetric_c4(args):
    result=summarize_sumo_replication(ROOT/"artifacts/farcal",ROOT/"artifacts/online",ROOT/"artifacts/predictions",
        ROOT/"artifacts/evaluation/sumo-asymmetric-C4-transfer.json",args.candidate,partition="test",method="far_cal_asymmetric",scenario="C4")
    print(json.dumps({key:result[key] for key in ("status","candidate_id","far_cal_interval_score","far_cal_coverage","primary_comparisons","superiority_rule_met")},indent=2))


def cmd_report_clean(args):
    source=ROOT/"artifacts"/"evaluation"/f"{args.dataset}-seed{args.seed}-clean.json"
    print(render_clean_report(source,ROOT/"reports"))


def cmd_evaluate_controls(args):
    prepared=ROOT/"data"/"prepared"/f"{args.dataset}.npz"
    output=ROOT/"artifacts"/"evaluation"/f"{args.dataset}-simple-controls-clean.json"
    print(json.dumps(evaluate_simple_controls(prepared,output),indent=2))


def cmd_sumo_build(args):
    config=load(args.config)
    spec=config["sumo"]
    output=ROOT / args.output
    manifest=build_network(output,spec["mainline_length_m"],spec["stations"],spec["through_lanes"])
    manifest["resolved_config_sha256"]=config_hash(config)
    print(json.dumps(manifest,indent=2))


def cmd_sumo_generate(args):
    config=load(args.config)
    spec=config["sumo"]
    network=ROOT / args.network
    output=ROOT / args.output
    generate_routes(network,output,spec["episode_seconds"],args.seed,args.family)
    sumocfg=write_config(network,output,spec["episode_seconds"],args.seed)
    events=json.loads((output/"episode_manifest.json").read_text(encoding="utf-8"))["physical_events"]
    hashes=run_offline(sumocfg,output,events)
    intervals,speeds,counts=parse_e1(output/"detectors.xml",spec["stations"],spec["through_lanes"])
    np.savez_compressed(output/"detector_truth.npz",intervals=intervals,speed_mph=speeds,nVehContrib=counts,original_valid=np.isfinite(speeds))
    summary={**hashes,"bins":len(intervals),"valid_fraction":float(np.isfinite(speeds).mean()),"units":"mph","seed":args.seed,"family":args.family}
    write_json(output/"run_summary.json",summary)
    print(json.dumps(summary,indent=2))


def cmd_sumo_paper(args):
    config=load(args.config)
    result=generate_paper_dataset(config,config_hash(config),ROOT/args.output,args.workers)
    print(json.dumps({"status":result["status"],"episodes":len(result["episodes"]),
                      "split_counts":result["split_counts"],"family_counts":result["family_counts"]},indent=2))


def cmd_sumo_replication(args):
    config=load(args.config); registry=json.loads((ROOT/args.registry).read_text(encoding="utf-8"))
    result=generate_replication_dataset(config,config_hash(config),ROOT/args.output,args.workers,registry)
    print(json.dumps({"status":result["status"],"episodes":len(result["episodes"]),"family_counts":result["family_counts"]},indent=2))


def cmd_smoke(args):
    config=load(args.config)
    base=ROOT/"data/sumo"
    network=base/"network"
    build_network(network,config["sumo"]["mainline_length_m"],config["sumo"]["stations"],config["sumo"]["through_lanes"])
    episode_paths=[]
    for role,seed in (("train",11),("calibration",22),("test",33)):
        episode=base/f"smoke-{role}-seed{seed}"
        truth=episode/"detector_truth.npz"
        if not truth.exists():
            generate_routes(network,episode,config["sumo"]["episode_seconds"],seed,"P0")
            sumocfg=write_config(network,episode,config["sumo"]["episode_seconds"],seed)
            run_offline(sumocfg,episode)
            intervals,speeds,counts=parse_e1(episode/"detectors.xml",20,2)
            np.savez_compressed(truth,intervals=intervals,speed_mph=speeds,nVehContrib=counts,original_valid=np.isfinite(speeds))
        episode_paths.append(truth)
    output=ROOT/"artifacts"/f"smoke-{config_hash(config)[:12]}"
    model,stats,training=train_smoke(episode_paths[0],network/"adjacency.npz",output,config["training"]["epochs"],config["training"]["seeds"][0])
    evaluation=smoke_calibration_eval(model,stats,episode_paths[1],episode_paths[2],network/"adjacency.npz",output)
    print(json.dumps({"training":training,"evaluation":evaluation},indent=2))


def parser():
    result=argparse.ArgumentParser(prog="python -m ps1.cli")
    sub=result.add_subparsers(dest="command",required=True)
    env=sub.add_parser("audit-environment"); env.set_defaults(func=cmd_audit_environment)
    audit=sub.add_parser("audit-data"); audit.add_argument("--dataset",choices=["metr_la","pems_bay"],required=True); audit.set_defaults(func=cmd_audit_data)
    prepare=sub.add_parser("prepare"); prepare.add_argument("--dataset",choices=["metr_la","pems_bay"],required=True); prepare.add_argument("--config"); prepare.set_defaults(func=cmd_prepare)
    faults=sub.add_parser("make-faults"); faults.add_argument("--dataset",choices=["metr_la","pems_bay"],required=True); faults.add_argument("--scenario",choices=[f"C{i}" for i in range(6)],required=True); faults.add_argument("--seed",type=int,required=True); faults.set_defaults(func=cmd_make_faults)
    registered=sub.add_parser("make-registered-faults"); registered.set_defaults(func=cmd_make_registered_faults)
    build=sub.add_parser("sumo-build"); build.add_argument("--config",default="configs/sumo.yaml"); build.add_argument("--output",default="data/sumo/network"); build.set_defaults(func=cmd_sumo_build)
    generate=sub.add_parser("sumo-generate"); generate.add_argument("--config",default="configs/smoke.yaml"); generate.add_argument("--network",default="data/sumo/network"); generate.add_argument("--output",default="data/sumo/smoke-p0-seed11"); generate.add_argument("--seed",type=int,default=11); generate.add_argument("--family",choices=["P0","P1","P2","P3"],default="P0"); generate.set_defaults(func=cmd_sumo_generate)
    paper=sub.add_parser("sumo-paper"); paper.add_argument("--config",default="configs/sumo.yaml"); paper.add_argument("--output",default="data/sumo/paper"); paper.add_argument("--workers",type=int,default=3); paper.set_defaults(func=cmd_sumo_paper)
    replication=sub.add_parser("sumo-replication"); replication.add_argument("--config",default="configs/sumo.yaml"); replication.add_argument("--registry",default="configs/sumo_independent_replication.json"); replication.add_argument("--output",default="data/sumo/replication"); replication.add_argument("--workers",type=int,default=3); replication.set_defaults(func=cmd_sumo_replication)
    smoke=sub.add_parser("smoke"); smoke.add_argument("--config",default="configs/smoke.yaml"); smoke.set_defaults(func=cmd_smoke)
    live=sub.add_parser("sumo-live"); live.add_argument("--config",default="configs/smoke.yaml"); live.add_argument("--seconds",type=int,default=900); live.add_argument("--seed",type=int,default=44); live.add_argument("--family",choices=["P0","P1","P2","P3"],default="P0"); live.set_defaults(func=cmd_sumo_live)
    report=sub.add_parser("report"); report.add_argument("--registry",default="configs/experiment_registry.yaml"); report.set_defaults(func=cmd_report)
    prepare_sumo=sub.add_parser("prepare-sumo"); prepare_sumo.add_argument("--input",default="data/sumo/paper"); prepare_sumo.add_argument("--output",default="data/prepared/sumo.npz"); prepare_sumo.set_defaults(func=cmd_prepare_sumo)
    prepare_replication=sub.add_parser("prepare-sumo-replication"); prepare_replication.add_argument("--input",default="data/sumo/replication"); prepare_replication.add_argument("--reference",default="data/prepared/sumo.npz"); prepare_replication.add_argument("--output",default="data/prepared/sumo_replication.npz"); prepare_replication.set_defaults(func=cmd_prepare_sumo_replication)
    sumo_faults=sub.add_parser("make-sumo-faults"); sumo_faults.add_argument("--scenario",choices=["C0","C1","C2","C3","C4"],required=True); sumo_faults.add_argument("--seed",type=int,required=True); sumo_faults.add_argument("--partition",choices=["tune","test","replication","confirmation","asym_validation"],default="test"); sumo_faults.set_defaults(func=cmd_make_sumo_faults)
    sumo_oracle=sub.add_parser("make-sumo-feedback-oracle"); sumo_oracle.add_argument("--seed",type=int,required=True); sumo_oracle.set_defaults(func=cmd_make_sumo_feedback_oracle)
    train=sub.add_parser("train"); train.add_argument("--dataset",choices=["metr_la","pems_bay","sumo"],required=True); train.add_argument("--seed",type=int,required=True); train.set_defaults(func=cmd_train)
    feasible=sub.add_parser("train-feasibility"); feasible.add_argument("--dataset",choices=["metr_la","pems_bay","sumo"],required=True); feasible.add_argument("--seed",type=int,default=11); feasible.set_defaults(func=cmd_train_feasibility)
    cache=sub.add_parser("cache-predictions"); cache.add_argument("--dataset",choices=["metr_la","pems_bay","sumo"],required=True); cache.add_argument("--seed",type=int,required=True); cache.add_argument("--partition",choices=["tune","calibration","test"],required=True); cache.set_defaults(func=cmd_cache_predictions)
    fault_cache=sub.add_parser("cache-faulted"); fault_cache.add_argument("--dataset",choices=["metr_la","pems_bay","sumo"],required=True); fault_cache.add_argument("--model-seed",type=int,required=True); fault_cache.add_argument("--scenario",choices=[f"C{i}" for i in range(6)],required=True); fault_cache.add_argument("--fault-seed",type=int,required=True); fault_cache.add_argument("--partition",choices=["tune","test","replication","confirmation","asym_validation"],default="test"); fault_cache.set_defaults(func=cmd_cache_faulted)
    validation=sub.add_parser("validate-c0-cache"); validation.add_argument("--dataset",choices=["metr_la","pems_bay","sumo"],required=True); validation.add_argument("--model-seed",type=int,required=True); validation.add_argument("--fault-seed",type=int,required=True); validation.set_defaults(func=cmd_validate_c0_cache)
    evaluate=sub.add_parser("evaluate-clean"); evaluate.add_argument("--dataset",choices=["metr_la","pems_bay","sumo"],required=True); evaluate.add_argument("--seed",type=int,required=True); evaluate.set_defaults(func=cmd_evaluate_clean)
    static_fault=sub.add_parser("evaluate-static-fault"); static_fault.add_argument("--dataset",choices=["metr_la","pems_bay","sumo"],required=True); static_fault.add_argument("--model-seed",type=int,required=True); static_fault.add_argument("--scenario",choices=[f"C{i}" for i in range(6)],required=True); static_fault.add_argument("--fault-seed",type=int,required=True); static_fault.set_defaults(func=cmd_evaluate_static_fault)
    online=sub.add_parser("evaluate-online"); online.add_argument("--dataset",choices=["metr_la","pems_bay","sumo"],required=True); online.add_argument("--model-seed",type=int,required=True); online.add_argument("--point-scenario",choices=[f"C{i}" for i in range(6)],required=True); online.add_argument("--feedback-scenario",choices=[*[f"C{i}" for i in range(6)],"C4I"],required=True); online.add_argument("--fault-seed",type=int,required=True); online.add_argument("--partition",choices=["tune","test","replication","confirmation","asym_validation"],default="test"); online.add_argument("--episode-subset"); online.set_defaults(func=cmd_evaluate_online)
    paired=sub.add_parser("compare-feedback"); paired.add_argument("--dataset",choices=["metr_la","pems_bay","sumo"],required=True); paired.add_argument("--model-seed",type=int,required=True); paired.add_argument("--point-scenario",choices=[f"C{i}" for i in range(6)],required=True); paired.add_argument("--immediate-scenario",choices=[*[f"C{i}" for i in range(6)],"C4I"],required=True); paired.add_argument("--delayed-scenario",choices=[f"C{i}" for i in range(6)],required=True); paired.add_argument("--fault-seed",type=int,required=True); paired.add_argument("--episode-subset"); paired.set_defaults(func=cmd_compare_feedback)
    input_mechanism=sub.add_parser("compare-sumo-input"); input_mechanism.add_argument("--model-seed",type=int,required=True); input_mechanism.add_argument("--fault-seed",type=int,default=101); input_mechanism.add_argument("--episode-subset",default="configs/sumo_mechanism_subset12.json"); input_mechanism.set_defaults(func=cmd_compare_sumo_input)
    recovery=sub.add_parser("evaluate-sumo-recovery"); recovery.add_argument("--model-seed",type=int,required=True); recovery.add_argument("--scenario",choices=["C3","C4"],required=True); recovery.add_argument("--fault-seed",type=int,required=True); recovery.set_defaults(func=cmd_evaluate_sumo_recovery)
    recovery_summary=sub.add_parser("summarize-sumo-recovery"); recovery_summary.set_defaults(func=cmd_summarize_sumo_recovery)
    farcal=sub.add_parser("evaluate-sumo-farcal"); farcal.add_argument("--model-seed",type=int,required=True); farcal.add_argument("--scenario",choices=["C3","C4"],required=True); farcal.add_argument("--fault-seed",type=int,required=True); farcal.add_argument("--partition",choices=["tune","test","replication","confirmation","asym_validation"],default="test"); farcal.add_argument("--episode-subset"); farcal.add_argument("--methods",default="far_cal,no_spatial,no_context,no_age,no_stale_blend,no_inflation"); farcal.add_argument("--tuning-candidate"); farcal.set_defaults(func=cmd_evaluate_sumo_farcal)
    real_farcal=sub.add_parser("evaluate-real-farcal"); real_farcal.add_argument("--dataset",choices=["metr_la","pems_bay"],required=True); real_farcal.add_argument("--model-seed",type=int,required=True); real_farcal.add_argument("--scenario",choices=["C3","C4"],default="C3"); real_farcal.add_argument("--fault-seed",type=int,required=True); real_farcal.set_defaults(func=cmd_evaluate_real_farcal)
    real_farcal_summary=sub.add_parser("summarize-real-transfer"); real_farcal_summary.add_argument("--dataset",choices=["pems_bay","metr_la"],required=True); real_farcal_summary.add_argument("--scenario",choices=["C3","C4"],default="C3"); real_farcal_summary.set_defaults(func=cmd_summarize_real_transfer)
    farcal_summary=sub.add_parser("summarize-sumo-farcal"); farcal_summary.set_defaults(func=cmd_summarize_sumo_farcal)
    locked_farcal_summary=sub.add_parser("summarize-locked-sumo-farcal"); locked_farcal_summary.add_argument("--candidate",default="n2b005w2"); locked_farcal_summary.set_defaults(func=cmd_summarize_locked_sumo_farcal)
    replication_summary=sub.add_parser("summarize-sumo-replication"); replication_summary.add_argument("--candidate",default="v2s25"); replication_summary.set_defaults(func=cmd_summarize_sumo_replication)
    confirmation_summary=sub.add_parser("summarize-sumo-confirmation"); confirmation_summary.add_argument("--candidate",default="v2s25"); confirmation_summary.set_defaults(func=cmd_summarize_sumo_confirmation)
    asymmetric_summary=sub.add_parser("summarize-asymmetric-development"); asymmetric_summary.add_argument("--candidate",default="asym100"); asymmetric_summary.set_defaults(func=cmd_summarize_asymmetric_development)
    asymmetric_validation=sub.add_parser("summarize-asymmetric-validation"); asymmetric_validation.add_argument("--candidate",default="asym100"); asymmetric_validation.set_defaults(func=cmd_summarize_asymmetric_validation)
    asymmetric_c4=sub.add_parser("summarize-asymmetric-c4"); asymmetric_c4.add_argument("--candidate",default="asym100"); asymmetric_c4.set_defaults(func=cmd_summarize_asymmetric_c4)
    feedback_report=sub.add_parser("report-feedback"); feedback_report.add_argument("--dataset",choices=["metr_la","pems_bay","sumo"],required=True); feedback_report.add_argument("--model-seed",type=int,required=True); feedback_report.add_argument("--point-scenario",choices=[f"C{i}" for i in range(6)],required=True); feedback_report.add_argument("--immediate-scenario",choices=[*[f"C{i}" for i in range(6)],"C4I"],required=True); feedback_report.add_argument("--delayed-scenario",choices=[f"C{i}" for i in range(6)],required=True); feedback_report.add_argument("--fault-seed",type=int,required=True); feedback_report.set_defaults(func=cmd_report_feedback)
    clean_report=sub.add_parser("report-clean"); clean_report.add_argument("--dataset",choices=["metr_la","pems_bay","sumo"],required=True); clean_report.add_argument("--seed",type=int,required=True); clean_report.set_defaults(func=cmd_report_clean)
    controls=sub.add_parser("evaluate-controls"); controls.add_argument("--dataset",choices=["metr_la","pems_bay"],required=True); controls.set_defaults(func=cmd_evaluate_controls)
    return result


def main(argv=None):
    args=parser().parse_args(argv); args.func(args)


if __name__=="__main__":
    main()
