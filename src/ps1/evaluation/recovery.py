import numpy as np
from pathlib import Path
import json

from ps1.evaluation.online_compare import _cache_arrays,_cache_manifest,_align_origins,_interval_arrays
from ps1.utils.artifacts import sha256_file,write_json


def recovery_summary(origin_bins,covered,interval_score,oracle_score,valid,t_restore,alpha,
                     window_origins=12,min_pairs=100,tolerance=.02,consecutive=3,
                     next_event=None,oracle_epsilon=1e-8):
    origins=np.asarray(origin_bins); covered=np.asarray(covered,bool); valid=np.asarray(valid,bool)
    scores=np.asarray(interval_score,float); oracle=np.asarray(oracle_score,float)
    if covered.shape!=valid.shape or scores.shape!=valid.shape or oracle.shape!=valid.shape or covered.shape[0]!=len(origins):
        raise ValueError("recovery arrays do not align")
    eligible=np.flatnonzero((origins>=t_restore)&((origins<next_event) if next_event is not None else True))
    windows=[]
    for position in range(window_origins-1,len(eligible)):
        rows=eligible[position-window_origins+1:position+1]
        mask=valid[rows]; count=int(mask.sum())
        if count:
            coverage=float(covered[rows][mask].mean()); score=float(scores[rows][mask].mean()); oracle_value=float(oracle[rows][mask].mean())
        else:
            coverage=score=oracle_value=float("nan")
        ratio=score/max(oracle_value,oracle_epsilon) if count else float("nan")
        windows.append({"end_bin":int(origins[rows[-1]]),"count":count,"coverage":coverage,
                        "interval_score":score,"oracle_interval_score":oracle_value,"score_ratio":ratio,
                        "oracle_epsilon_used":bool(count and oracle_value<=oracle_epsilon)})
    threshold=1-alpha-tolerance
    def first(require_joint):
        for i in range(max(0,len(windows)-consecutive+1)):
            sample=windows[i:i+consecutive]
            good=all(row["count"]>=min_pairs and row["coverage"]>=threshold and
                     ((row["score_ratio"]<=1.10) if require_joint else True) for row in sample)
            if good: return sample[0]["end_bin"]-t_restore
        return None
    return {"t_restore":t_restore,"minimum_measurable_delay_bins":window_origins-1,
            "coverage_recovery_bins":first(False),"joint_recovery_bins":first(True),
            "coverage_censored":first(False) is None,"joint_censored":first(True) is None,
            "windows":windows}


def evaluate_sumo_recovery(candidate_dir,oracle_dir,test_cache_dir,fault_dir,output_path,
                           window_origins=12,min_pairs=100,tolerance=.02,consecutive=3):
    candidate,candidate_manifest=_cache_manifest(candidate_dir); oracle,oracle_manifest=_cache_manifest(oracle_dir)
    test,test_manifest=_cache_manifest(test_cache_dir); fault_path=Path(fault_dir)/"fault_manifest.json"
    fault=json.loads(fault_path.read_text(encoding="utf-8"))
    for key in ("dataset","model_seed","point_scenario","point_fault_seed"):
        if candidate.get(key)!=oracle.get(key): raise ValueError(f"recovery arms differ on {key}")
    if candidate["dataset"]!="sumo" or not candidate.get("episode_reset"): raise ValueError("SUMO episode-reset online artifacts required")
    ca=_cache_arrays(candidate_dir,candidate); oa=_cache_arrays(oracle_dir,oracle)
    if not np.array_equal(ca["issue_bin"],oa["issue_bin"]): raise ValueError("recovery issue bins differ")
    ta=_align_origins(_cache_arrays(test_cache_dir,test),ca["issue_bin"])
    episodes=ta["episode_index"]; bins_per_episode=candidate["bins_per_episode"]; records=[]
    for event in fault["events"]:
        episode=int(event["episode_index"]); rows=np.flatnonzero(episodes==episode)
        if not len(rows): continue
        local_origins=ta["issue_bin"][rows]-episode*bins_per_episode
        sensors=np.asarray(event["affected_sensors"],dtype=int); t_restore=int(event["local_end_bin"])
        for method_index,method in enumerate(candidate["methods"]):
            for level_index,level in enumerate(candidate["levels"]):
                for horizon_index,horizon in enumerate(candidate["report_horizons"]):
                    truth=ta["truth_mph"][rows,horizon-1][:,sensors]; valid=ta["original_valid"][rows,horizon-1][:,sensors]
                    cvalues=_interval_arrays(truth,ca["lower_mph"][rows,method_index,level_index,horizon_index][:,sensors],
                        ca["upper_mph"][rows,method_index,level_index,horizon_index][:,sensors],valid,1-level)
                    ovalues=_interval_arrays(truth,oa["lower_mph"][rows,method_index,level_index,horizon_index][:,sensors],
                        oa["upper_mph"][rows,method_index,level_index,horizon_index][:,sensors],valid,1-level)
                    summary=recovery_summary(local_origins,cvalues[1],cvalues[0],ovalues[0],valid,t_restore,1-level,
                        window_origins,min_pairs,tolerance,consecutive)
                    records.append({"episode_index":episode,"event_index":event["event_index"],"method":method,
                        "level":level,"horizon":horizon,"affected_sensors":sensors.tolist(),**summary})
    result={"schema_version":1,"status":"completed","dataset":"sumo","scenario":fault["scenario_id"],
        "model_seed":candidate["model_seed"],"fault_seed":fault["seed"],"oracle_feedback_scenario":oracle["feedback_scenario"],
        "definition":{"window_origins":window_origins,"min_pairs":min_pairs,"tolerance":tolerance,"consecutive_windows":consecutive},
        "records":records,"record_count":len(records),
        "coverage_censored_count":sum(row["coverage_censored"] for row in records),
        "joint_censored_count":sum(row["joint_censored"] for row in records),
        "candidate_manifest_sha256":sha256_file(candidate_manifest),"oracle_manifest_sha256":sha256_file(oracle_manifest),
        "test_manifest_sha256":sha256_file(test_manifest),"fault_manifest_sha256":sha256_file(fault_path)}
    write_json(output_path,result); return result


def summarize_sumo_recovery(evaluation_dir,output_path,model_seeds=(11,22,33),fault_seeds=(101,202,303)):
    sources=[]; total_records=coverage_censored=joint_censored=0; maximum_window_pairs=0
    for model_seed in model_seeds:
        for scenario in ("C3","C4"):
            for fault_seed in fault_seeds:
                path=Path(evaluation_dir)/f"sumo-seed{model_seed}-{scenario}-recovery-fault{fault_seed}.json"
                result=json.loads(path.read_text(encoding="utf-8"))
                if result.get("status")!="completed": raise ValueError(f"incomplete recovery artifact: {path}")
                total_records+=result["record_count"]; coverage_censored+=result["coverage_censored_count"]
                joint_censored+=result["joint_censored_count"]
                maximum_window_pairs=max(maximum_window_pairs,max((window["count"] for row in result["records"] for window in row["windows"]),default=0))
                sources.append({"file":path.name,"sha256":sha256_file(path)})
    summary={"schema_version":1,"status":"completed","dataset":"sumo","runs":len(sources),"model_seeds":list(model_seeds),
        "fault_seeds":list(fault_seeds),"scenarios":["C3","C4"],"record_count":total_records,
        "coverage_censored_count":coverage_censored,"joint_censored_count":joint_censored,
        "maximum_window_pairs":maximum_window_pairs,"registered_minimum_pairs":100,
        "interpretation":"Recovery time is not identifiable under the registered per-event/per-horizon rule because every 12-origin window over two affected sensors has at most 24 valid pairs.",
        "sources":sources}
    write_json(output_path,summary); return summary
