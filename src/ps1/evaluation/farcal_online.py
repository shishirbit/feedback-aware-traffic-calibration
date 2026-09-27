"""Episode-reset exact-array FAR-Cal and registered ablations for SUMO."""
from collections import defaultdict
from pathlib import Path
import json

import numpy as np

from ps1.calibration.frozen import weighted_quantile
from ps1.data.graphs import local_groups
from ps1.evaluation.bootstrap import episode_mean_ci,moving_block_mean_ci
from ps1.evaluation.online_compare import LEVELS,OnlineMetricSums,_arrival_order,_cache_arrays,_cache_manifest,_align_origins,_interval_arrays
from ps1.utils.artifacts import sha256_file,write_json


METHODS=("far_cal","far_cal_asymmetric","no_spatial","no_context","no_age","no_stale_blend","no_inflation")


def _pool(records,mask,now,context,tau,bandwidth,use_age=True,use_context=True,freshness="target"):
    scores=records["score"][mask]
    if not len(scores): return scores,np.empty(0),0.
    targets=records["target"][mask]
    log_weights=np.zeros(len(scores),float)
    if use_age: log_weights-=((now-records[freshness][mask])/tau)
    if use_context:
        difference=records["context"][mask]-context
        log_weights-=np.square(difference).sum(axis=1)/(2*bandwidth**2)
    weights=np.exp(log_weights-log_weights.max())
    _,bins=np.unique(targets,return_inverse=True); blocks=np.bincount(bins,weights=weights)
    ess=float(blocks.sum()**2/np.square(blocks).sum())
    return scores,weights/weights.sum(),ess


def _method_distribution(method,records,frozen,now,sensor,context,missing_fraction,groups,parameters):
    if method == "far_cal_asymmetric":
        ablation = parameters.get("ablation", "full")
        if ablation not in {"full", "no_spatial", "no_context", "no_age", "no_stale_blend", "no_inflation", "arrival_freshness", "frozen_only"}:
            raise ValueError(f"unknown asymmetric ablation: {ablation}")
        method = ablation
    freshness = "arrival" if method == "arrival_freshness" else "target"
    use_age=method!="no_age"; use_context=method!="no_context"
    group=(sensor,) if method=="no_spatial" else groups[sensor]
    local_mask=np.isin(records["sensor"],group); global_mask=np.ones(len(records["score"]),bool)
    if parameters.get("scalable_local_only"):
        local_indices=np.flatnonzero(local_mask); cap=int(parameters["records_per_sensor_cap"])*len(group)
        if len(local_indices)>cap:
            local_mask=np.zeros(len(records["score"]),bool); local_mask[local_indices[-cap:]]=True
        ls,lw,le=_pool(records,local_mask,now,context,parameters["tau"],parameters["bandwidth"],use_age,use_context,freshness)
        frozen_mask=np.isin(frozen["sensor"],group); frozen_indices=np.flatnonzero(frozen_mask)
        if len(frozen_indices)>cap: frozen_indices=frozen_indices[-cap:]
        frozen_scores=frozen["score"][frozen_indices]
        if not len(frozen_scores): frozen_scores=frozen["score"][-cap:]
        freshest=records[freshness][local_mask] if local_mask.any() else frozen[freshness][frozen_indices]
        gap=max(0,now-int(freshest.max())) if len(freshest) else parameters["buffer_bins"]
        if method == "frozen_only":
            return frozen_scores, np.full(len(frozen_scores),1/len(frozen_scores)), 1.
        eta=(1. if method=="no_stale_blend" else min(1.,le/parameters["support"])*np.exp(-gap/parameters["stale_decay"])) if len(ls) else 0.
        scores=np.concatenate((ls,frozen_scores)); weights=np.concatenate((eta*lw,np.full(len(frozen_scores),(1-eta)/len(frozen_scores))))
        inflation=1. if method=="no_inflation" else min(parameters["inflation_cap"],1+parameters["beta_gap"]*min(gap/parameters["stale_decay"],2)+parameters["beta_missing"]*missing_fraction)
        return scores,weights,inflation
    ls,lw,le=_pool(records,local_mask,now,context,parameters["tau"],parameters["bandwidth"],use_age,use_context,freshness)
    gs,gw,ge=_pool(records,global_mask,now,context,parameters["tau"],parameters["bandwidth"],use_age,use_context,freshness)
    frozen_node=frozen["sensor"]==sensor; frozen_scores=frozen["score"][frozen_node]
    if len(frozen_scores)<50: frozen_scores=frozen["score"]
    if method == "frozen_only":
        return frozen_scores, np.full(len(frozen_scores),1/len(frozen_scores)), 1.
    archive_group=np.isin(frozen["sensor"],group)
    freshest_targets=records[freshness][local_mask] if local_mask.any() else frozen[freshness][archive_group]
    if not len(freshest_targets): freshest_targets=frozen[freshness]
    gap=max(0,now-int(freshest_targets.max()))
    eta=(1. if method=="no_stale_blend" else min(1.,ge/parameters["support"])*np.exp(-gap/parameters["stale_decay"])) if len(gs) else 0.
    mix=le/(le+parameters["support"]) if len(ls) else 0.
    scores=np.concatenate((ls,gs,frozen_scores))
    weights=np.concatenate((eta*mix*lw,eta*(1-mix)*gw,np.full(len(frozen_scores),(1-eta)/len(frozen_scores))))
    inflation=1. if method=="no_inflation" else min(parameters["inflation_cap"],1+parameters["beta_gap"]*min(gap/parameters["stale_decay"],2)+parameters["beta_missing"]*missing_fraction)
    shrinkage=parameters.get("rolling_shrinkage",0.) if method=="far_cal" else 0.
    if shrinkage:
        rolling_scores=records["score"][records["sensor"]==sensor]
        if len(rolling_scores)<50: rolling_scores=records["score"]
        scores=np.concatenate((scores,rolling_scores)); weights=np.concatenate(((1-shrinkage)*weights,np.full(len(rolling_scores),shrinkage/len(rolling_scores))))
        inflation=1+(1-shrinkage)*(inflation-1)
    return scores,weights,inflation


def _weighted_levels(scores,weights,order=None):
    return _weighted_probabilities(scores,weights,LEVELS,order)


def _weighted_probabilities(scores,weights,probabilities,order=None):
    keep=weights>0; scores=scores[keep]; weights=weights[keep]
    if order is None or not keep.all(): order=np.argsort(scores,kind="stable")
    cumulative=np.cumsum(weights[order]); total=cumulative[-1]
    return tuple(float(scores[order[min(np.searchsorted(cumulative,level*total,side="left"),len(scores)-1)]]) for level in probabilities)


def _method_widths(method,records,frozen,now,sensor,horizon,context,missing_fraction,groups,parameters):
    del horizon
    scores,weights,inflation=_method_distribution(method,records,frozen,now,sensor,context,missing_fraction,groups,parameters)
    return tuple(value*inflation for value in _weighted_levels(scores,weights))


def _calibration_records(calibration_dir,manifest,horizons,bins_per_episode=96,episode_subset=None):
    arrays=_cache_arrays(calibration_dir,manifest); episodes=arrays["episode_index"]; unique=np.unique(episodes)
    if episode_subset is not None:
        keep=np.isin(episodes,episode_subset); arrays={key:value[keep] if isinstance(value,np.ndarray) and len(value)==len(episodes) else value for key,value in arrays.items()}
        episodes=arrays["episode_index"]; unique=np.unique(episodes)
        if not len(unique): raise ValueError("calibration episode subset is empty")
    rank={int(value):index for index,value in enumerate(unique)}; mapped=np.asarray([rank[int(ep)]*288+(int(issue)-int(ep)*bins_per_episode) for ep,issue in zip(episodes,arrays["issue_bin"])])
    scale=np.maximum((arrays["q95_mph"]-arrays["q05_mph"])/2,1.); signed=(arrays["truth_mph"]-arrays["q50_mph"])/scale; score=np.abs(signed)
    records={}
    for horizon in horizons:
        valid=arrays["original_valid"][:,horizon-1]; row,sensor=np.nonzero(valid)
        records[horizon]={"score":score[row,horizon-1,sensor],"signed":signed[row,horizon-1,sensor],"target":mapped[row]+horizon,"sensor":sensor,
            "context":arrays["context"][row,sensor]}
    return records,len(unique)*288


def _continuous_calibration_records(calibration_dir,manifest,horizons,calibration_fault_dir=None,available_by=None):
    arrays=_cache_arrays(calibration_dir,manifest)
    scale=np.maximum((arrays["q95_mph"]-arrays["q05_mph"])/2,1.)
    signed=(arrays["truth_mph"]-arrays["q50_mph"])/scale; score=np.abs(signed)
    calibration_arrival=None
    if calibration_fault_dir is not None:
        calibration_fault_dir=Path(calibration_fault_dir)
        cal_fault=json.loads((calibration_fault_dir/"fault_manifest.json").read_text(encoding="utf-8"))
        with np.load(calibration_fault_dir/"release_schedule.npz") as schedule:
            calibration_arrival=schedule["feedback_arrival"].copy()
        calibration_start=int(cal_fault["partition_start_row"])
    records={}
    for horizon in horizons:
        valid=arrays["original_valid"][:,horizon-1].copy()
        if calibration_arrival is not None:
            local=arrays["issue_bin"]+horizon-calibration_start
            received=calibration_arrival[local]
            valid &= (received>=0)&(received<=available_by-calibration_start)
        row,sensor=np.nonzero(valid)
        records[horizon]={"score":score[row,horizon-1,sensor],"signed":signed[row,horizon-1,sensor],
            "target":arrays["issue_bin"][row]+horizon,"sensor":sensor,"context":arrays["context"][row,sensor]}
    return records


def _append(records,values):
    for key in records: records[key]=np.concatenate((records[key],np.asarray(values[key],dtype=records[key].dtype)))


def evaluate_sumo_farcal(calibration_dir,test_cache_dir,fault_dir,output_dir,config,report_horizons=(3,6,12),episode_subset=None,methods=METHODS,
        calibration_episode_subset=None):
    calibration,calibration_manifest=_cache_manifest(calibration_dir); test,test_manifest=_cache_manifest(test_cache_dir)
    fault_dir,output=Path(fault_dir),Path(output_dir); fault_path=fault_dir/"fault_manifest.json"
    fault=json.loads(fault_path.read_text(encoding="utf-8"))
    if not test["dataset"].startswith("sumo") or not test.get("episode_aware"): raise ValueError("episode-aware SUMO cache required")
    if test["scenario"]!=fault["scenario_id"] or test["release_schedule_sha256"]!=sha256_file(fault_dir/"release_schedule.npz"): raise ValueError("point cache and fault schedule differ")
    arrays=_cache_arrays(test_cache_dir,test); episodes=arrays["episode_index"]
    with np.load(fault_dir/"release_schedule.npz") as schedule: feedback_arrival=schedule["feedback_arrival"].copy()
    parameters=config["parameters"]
    with np.load(config["prepared_path"]) as prepared: groups=local_groups(prepared["adjacency"],parameters.get("maximum_neighbors",4))
    frozen,test_base=_calibration_records(calibration_dir,calibration,report_horizons,episode_subset=calibration_episode_subset)
    if parameters.get("ablation") == "arrival_freshness":
        for record in frozen.values(): record["arrival"] = record["target"].copy()
    bins_per_episode=fault["test_length"]//len(fault["test_episode_indices"])
    context_mean=np.asarray(test["context_mean"]); context_std=np.asarray(test["context_std"])
    methods=tuple(methods)
    if not methods or not set(methods)<=set(METHODS): raise ValueError("unknown or empty FAR-Cal method set")
    selected=set(fault["test_episode_indices"] if episode_subset is None else episode_subset)
    if not selected<=set(fault["test_episode_indices"]): raise ValueError("episode subset contains non-test episode")
    metrics={(method,level,h):OnlineMetricSums() for method in methods for level in LEVELS for h in report_horizons}
    chunks=[]; output.mkdir(parents=True,exist_ok=True)
    for source_ordinal,episode in enumerate(fault["test_episode_indices"]):
        if episode not in selected: continue
        rows=np.flatnonzero(episodes==episode); issue=arrays["issue_bin"][rows]; local_issue=issue-(fault["test_start_row"]+source_ordinal*bins_per_episode)
        now0=test_base+int(local_issue[0]); live={h:{key:value.copy() for key,value in frozen[h].items()} for h in report_horizons}
        for h in report_horizons:
            keep=live[h]["target"]>=now0-parameters["buffer_bins"]
            live[h]={key:value[keep] for key,value in live[h].items()}
        start=source_ordinal*bins_per_episode; local_feedback=np.where(feedback_arrival[start:start+bins_per_episode]>=0,feedback_arrival[start:start+bins_per_episode]-start,-1)
        arrival,observed,arrival_sensor=_arrival_order(local_feedback,int(local_issue[-1])); pointer=0; pending={}
        lower=np.empty((len(rows),len(methods),len(LEVELS),len(report_horizons),test["sensors"]),np.float32); upper=np.empty_like(lower)
        point=arrays["q50_mph"][rows]; scale=np.maximum((arrays["q95_mph"][rows]-arrays["q05_mph"][rows])/2,1.); truth=arrays["truth_mph"][rows]; valid=arrays["original_valid"][rows]; contexts=arrays["context"][rows]
        for row,local in enumerate(local_issue):
            local=int(local); now=test_base+local; additions={h:defaultdict(list) for h in report_horizons}
            end=np.searchsorted(arrival,local,side="right")
            for index in range(pointer,end):
                target=int(observed[index]); sensor=int(arrival_sensor[index])
                for horizon in report_horizons:
                    record=pending.get((target,horizon))
                    if record is None or not record["valid"][sensor]: continue
                    signed=(record["truth"][sensor]-record["point"][sensor])/record["scale"][sensor]
                    additions[horizon]["score"].append(abs(signed)); additions[horizon]["signed"].append(signed)
                    if "arrival" in live[horizon]: additions[horizon]["arrival"].append(test_base+int(arrival[index]))
                    additions[horizon]["target"].append(test_base+target); additions[horizon]["sensor"].append(sensor); additions[horizon]["context"].append(record["context"][sensor])
            pointer=end
            for horizon in report_horizons:
                if additions[horizon]: _append(live[horizon],additions[horizon])
                keep=live[horizon]["target"]>=now-parameters["buffer_bins"]
                live[horizon]={key:value[keep] for key,value in live[horizon].items()}
                hi=horizon-1
                for sensor in range(test["sensors"]):
                    missing=1-(contexts[row,sensor,0]*context_std[0]+context_mean[0]); missing=float(np.clip(missing,0,1))
                    distributions={}
                    for method in methods:
                        if method=="no_inflation" and "far_cal" in distributions:
                            scores,weights,_=distributions["far_cal"]; distributions[method]=(scores,weights,1.)
                        elif method=="far_cal_asymmetric":
                            signed_live={**live[horizon],"score":live[horizon]["signed"]}; signed_frozen={**frozen[horizon],"score":frozen[horizon]["signed"]}
                            distributions[method]=_method_distribution(method,signed_live,signed_frozen,now,sensor,contexts[row,sensor],missing,groups,parameters)
                        else:
                            distributions[method]=_method_distribution(method,live[horizon],frozen[horizon],now,sensor,contexts[row,sensor],missing,groups,parameters)
                    common_method=next((method for method in methods if method!="no_spatial"),None)
                    common_order=np.argsort(distributions[common_method][0],kind="stable") if common_method else None
                    for method_index,method in enumerate(methods):
                        scores,weights,inflation=distributions[method]
                        order=None if method=="no_spatial" or common_order is None or len(common_order)!=len(scores) else common_order
                        if method=="far_cal_asymmetric":
                            probabilities=tuple(value for level in LEVELS for value in ((1-level)/2,1-(1-level)/2))
                            signed_factors=_weighted_probabilities(scores,weights,probabilities,order)
                            rolling_mask=live[horizon]["sensor"]==sensor; rolling_scores=live[horizon]["score"][rolling_mask]
                            if len(rolling_scores)<50: rolling_scores=live[horizon]["score"]
                            rolling_factors=_weighted_levels(rolling_scores,np.full(len(rolling_scores),1/len(rolling_scores)))
                            blend=parameters["asymmetric_blend"]
                            factors=[((1-blend)*-rolling_factors[i]+blend*inflation*signed_factors[2*i],
                                      (1-blend)*rolling_factors[i]+blend*inflation*signed_factors[2*i+1]) for i in range(len(LEVELS))]
                        else:
                            symmetric=tuple(value*inflation for value in _weighted_levels(scores,weights,order)); factors=[(-value,value) for value in symmetric]
                        for level_index,(level,(lower_factor,upper_factor)) in enumerate(zip(LEVELS,factors)):
                            lo=max(0,point[row,hi,sensor]+scale[row,hi,sensor]*lower_factor); up=max(lo,point[row,hi,sensor]+scale[row,hi,sensor]*upper_factor)
                            lower[row,method_index,level_index,report_horizons.index(horizon),sensor]=lo; upper[row,method_index,level_index,report_horizons.index(horizon),sensor]=up
                pending[(local+horizon,horizon)]={"point":point[row,hi].copy(),"scale":scale[row,hi].copy(),"truth":truth[row,hi].copy(),"valid":valid[row,hi].copy(),"context":contexts[row].copy()}
                for method_index,method in enumerate(methods):
                    for level_index,level in enumerate(LEVELS):
                        metrics[(method,level,horizon)].add(truth[row,hi],lower[row,method_index,level_index,report_horizons.index(horizon)],upper[row,method_index,level_index,report_horizons.index(horizon)],valid[row,hi],1-level)
        path=output/f"episode-{source_ordinal:02d}.npz"; np.savez_compressed(path,issue_bin=issue,episode_index=np.full(len(rows),episode,np.int16),lower_mph=lower,upper_mph=upper)
        chunks.append({"file":path.name,"sha256":sha256_file(path),"origins":len(rows),"episode_index":int(episode)})
    result={"schema_version":1,"status":"completed","dataset":"sumo","model_seed":test["checkpoint_training_seed"],"scenario":fault["scenario_id"],"fault_seed":fault["seed"],
        "episode_reset":True,"episode_count":len(chunks),"episode_indices":[chunk["episode_index"] for chunk in chunks],"episode_subset":episode_subset is not None,
        "methods":list(methods),"levels":list(LEVELS),"report_horizons":list(report_horizons),
        "candidate_id":config.get("candidate_id"),"calibration_episode_indices":None if calibration_episode_subset is None else list(calibration_episode_subset),
        "parameters":parameters,"metrics":{method:{str(level):{str(h):metrics[(method,level,h)].result() for h in report_horizons} for level in LEVELS} for method in methods},
        "calibration_manifest_sha256":sha256_file(calibration_manifest),"test_manifest_sha256":sha256_file(test_manifest),"fault_manifest_sha256":sha256_file(fault_path),"chunks":chunks}
    write_json(output/"manifest.json",result); return result


def evaluate_continuous_farcal(calibration_dir,test_cache_dir,fault_dir,output_dir,config,report_horizons=(3,6,12),methods=("far_cal_asymmetric",)):
    """Evaluate FAR-Cal on a chronological real-data test stream without episode resets."""
    calibration,calibration_manifest=_cache_manifest(calibration_dir); test,test_manifest=_cache_manifest(test_cache_dir)
    fault_dir,output=Path(fault_dir),Path(output_dir); fault_path=fault_dir/"fault_manifest.json"
    fault=json.loads(fault_path.read_text(encoding="utf-8"))
    if test.get("episode_aware") or test["dataset"].startswith("sumo"): raise ValueError("continuous real-data cache required")
    if test["scenario"]!=fault["scenario_id"] or test["release_schedule_sha256"]!=sha256_file(fault_dir/"release_schedule.npz"):
        raise ValueError("point cache and fault schedule differ")
    if calibration["checkpoint_sha256"]!=test["checkpoint_sha256"]: raise ValueError("calibration and test checkpoints differ")
    arrays=_cache_arrays(test_cache_dir,test); stride=int(config.get("evaluation_stride",1))
    if stride<1: raise ValueError("evaluation_stride must be positive")
    if stride>1: arrays={key:value[::stride] for key,value in arrays.items()}
    issue_bins=arrays["issue_bin"]
    with np.load(fault_dir/"release_schedule.npz") as schedule: feedback_arrival=schedule["feedback_arrival"].copy()
    parameters=config["parameters"]
    with np.load(config["prepared_path"]) as prepared: groups=local_groups(prepared["adjacency"],parameters.get("maximum_neighbors",4))
    frozen=_continuous_calibration_records(calibration_dir,calibration,report_horizons,
        config.get("calibration_fault_dir"),int(issue_bins[0]))
    if parameters.get("ablation") == "arrival_freshness":
        for record in frozen.values(): record["arrival"] = record["target"].copy()
    live={h:{key:value.copy() for key,value in frozen[h].items()} for h in report_horizons}
    first_issue=int(issue_bins[0]); test_start=int(fault["test_start_row"])
    for horizon in report_horizons:
        keep=live[horizon]["target"]>=first_issue-parameters["buffer_bins"]
        live[horizon]={key:value[keep] for key,value in live[horizon].items()}
    context_mean=np.asarray(test["context_mean"]); context_std=np.asarray(test["context_std"]); methods=tuple(methods)
    if not methods or not set(methods)<=set(METHODS): raise ValueError("unknown or empty FAR-Cal method set")
    max_local=int(issue_bins[-1])-test_start; arrival,observed,arrival_sensor=_arrival_order(feedback_arrival,max_local); pointer=0; pending={}
    metrics={(method,level,h):OnlineMetricSums() for method in methods for level in LEVELS for h in report_horizons}
    chunks=[]; output.mkdir(parents=True,exist_ok=True); offset=0
    source_chunks=test["chunks"] if stride==1 else [{"origins":len(issue_bins)}]
    for chunk_ordinal,chunk_meta in enumerate(source_chunks):
        count=chunk_meta["origins"]; rows=slice(offset,offset+count); offset+=count
        issue=arrays["issue_bin"][rows]; point=arrays["q50_mph"][rows]
        scale=np.maximum((arrays["q95_mph"][rows]-arrays["q05_mph"][rows])/2,1.)
        truth=arrays["truth_mph"][rows]; valid=arrays["original_valid"][rows]; contexts=arrays["context"][rows]
        lower=np.full((count,len(methods),len(LEVELS),len(report_horizons),test["sensors"]),np.nan,np.float32); upper=np.full_like(lower,np.nan)
        evaluation_mask=np.zeros((count,test["sensors"]),bool)
        for row,issue_value in enumerate(issue):
            now=int(issue_value); local=now-test_start; additions={h:defaultdict(list) for h in report_horizons}
            end=np.searchsorted(arrival,local,side="right")
            for index in range(pointer,end):
                target=test_start+int(observed[index]); sensor=int(arrival_sensor[index])
                for horizon in report_horizons:
                    record=pending.get((target,horizon))
                    if record is None or not record["valid"][sensor]: continue
                    signed=(record["truth"][sensor]-record["point"][sensor])/record["scale"][sensor]
                    additions[horizon]["score"].append(abs(signed)); additions[horizon]["signed"].append(signed)
                    if "arrival" in live[horizon]: additions[horizon]["arrival"].append(test_start+int(arrival[index]))
                    additions[horizon]["target"].append(target); additions[horizon]["sensor"].append(sensor); additions[horizon]["context"].append(record["context"][sensor])
            pointer=end
            if config.get("focus_fault_events"):
                recovery=int(config.get("focus_recovery_bins",0)); focus=set()
                for event in fault["events"]:
                    start, stop = int(event["start_bin"]), int(event["end_bin"])
                    if not start <= local <= stop + recovery:
                        continue
                    if "affected_sensors_by_bin" in event:
                        # The matched control changes failed identities every bin.
                        # Focus on entries failed now or within the recovery window.
                        by_bin = event["affected_sensors_by_bin"]
                        for affected_bin in range(max(start, local - recovery), min(stop, local + 1)):
                            focus.update(by_bin[affected_bin - start])
                    else:
                        focus.update(event["affected_sensors"])
                selected_sensors=sorted(focus); evaluation_mask[row,selected_sensors]=True
            else:
                selected_sensors=range(test["sensors"]); evaluation_mask[row]=True
            for horizon_index,horizon in enumerate(report_horizons):
                if additions[horizon]: _append(live[horizon],additions[horizon])
                keep=live[horizon]["target"]>=now-parameters["buffer_bins"]
                live[horizon]={key:value[keep] for key,value in live[horizon].items()}
                hi=horizon-1
                for sensor in selected_sensors:
                    missing=float(np.clip(1-(contexts[row,sensor,0]*context_std[0]+context_mean[0]),0,1)); distributions={}
                    for method in methods:
                        if method=="far_cal_asymmetric":
                            signed_live={**live[horizon],"score":live[horizon]["signed"]}; signed_frozen={**frozen[horizon],"score":frozen[horizon]["signed"]}
                            distributions[method]=_method_distribution(method,signed_live,signed_frozen,now,sensor,contexts[row,sensor],missing,groups,parameters)
                        else: distributions[method]=_method_distribution(method,live[horizon],frozen[horizon],now,sensor,contexts[row,sensor],missing,groups,parameters)
                    for method_index,method in enumerate(methods):
                        scores,weights,inflation=distributions[method]
                        if method=="far_cal_asymmetric":
                            probabilities=tuple(value for level in LEVELS for value in ((1-level)/2,1-(1-level)/2))
                            signed_factors=_weighted_probabilities(scores,weights,probabilities)
                            rolling_mask=live[horizon]["sensor"]==sensor; rolling_scores=live[horizon]["score"][rolling_mask]
                            if len(rolling_scores)<50: rolling_scores=live[horizon]["score"]
                            rolling_factors=_weighted_levels(rolling_scores,np.full(len(rolling_scores),1/len(rolling_scores)))
                            blend=parameters["asymmetric_blend"]
                            factors=[((1-blend)*-rolling_factors[i]+blend*inflation*signed_factors[2*i],(1-blend)*rolling_factors[i]+blend*inflation*signed_factors[2*i+1]) for i in range(len(LEVELS))]
                        else:
                            symmetric=tuple(value*inflation for value in _weighted_levels(scores,weights)); factors=[(-value,value) for value in symmetric]
                        for level_index,(level,(lower_factor,upper_factor)) in enumerate(zip(LEVELS,factors)):
                            lo=max(0,point[row,hi,sensor]+scale[row,hi,sensor]*lower_factor); up=max(lo,point[row,hi,sensor]+scale[row,hi,sensor]*upper_factor)
                            lower[row,method_index,level_index,horizon_index,sensor]=lo; upper[row,method_index,level_index,horizon_index,sensor]=up
                pending[(now+horizon,horizon)]={"point":point[row,hi].copy(),"scale":scale[row,hi].copy(),"truth":truth[row,hi].copy(),"valid":valid[row,hi].copy(),"context":contexts[row].copy()}
                for method_index,method in enumerate(methods):
                    for level_index,level in enumerate(LEVELS): metrics[(method,level,horizon)].add(truth[row,hi],lower[row,method_index,level_index,horizon_index],upper[row,method_index,level_index,horizon_index],valid[row,hi]&evaluation_mask[row],1-level)
        path=output/f"chunk-{chunk_ordinal:04d}.npz"; np.savez_compressed(path,issue_bin=issue,lower_mph=lower,upper_mph=upper,evaluation_mask=evaluation_mask)
        chunks.append({"file":path.name,"sha256":sha256_file(path),"origins":len(issue),"first_issue_bin":int(issue[0]),"last_issue_bin":int(issue[-1])})
    result={"schema_version":1,"status":"completed","dataset":test["dataset"],"model_seed":test["checkpoint_training_seed"],"scenario":fault["scenario_id"],"fault_seed":fault["seed"],
        "episode_reset":False,"evaluation_stride":stride,"focus_fault_events":bool(config.get("focus_fault_events")),"focus_recovery_bins":int(config.get("focus_recovery_bins",0)),"methods":list(methods),"levels":list(LEVELS),"report_horizons":list(report_horizons),"candidate_id":config.get("candidate_id"),"parameters":parameters,
        "metrics":{method:{str(level):{str(h):metrics[(method,level,h)].result() for h in report_horizons} for level in LEVELS} for method in methods},
        "calibration_manifest_sha256":sha256_file(calibration_manifest),"test_manifest_sha256":sha256_file(test_manifest),"fault_manifest_sha256":sha256_file(fault_path),
        "calibration_fault_manifest_sha256":sha256_file(Path(config["calibration_fault_dir"])/"fault_manifest.json") if config.get("calibration_fault_dir") else None,"chunks":chunks}
    write_json(output/"manifest.json",result); return result


def _stratified_block_ci(strata,block_bins=24,resamples=2000,seed=0):
    summaries=[]
    for values in strata:
        flat=np.asarray(values,float).reshape(len(values),-1); summaries.append((np.nansum(flat,axis=1),np.isfinite(flat).sum(axis=1)))
    total_sum=sum(x.sum() for x,_ in summaries); total_count=sum(x.sum() for _,x in summaries); rng=np.random.default_rng(seed); draws=[]
    for _ in range(resamples):
        draw_sum=0.; draw_count=0
        for sums,counts in summaries:
            starts=np.arange(len(sums)-block_bins+1); blocks=int(np.ceil(len(sums)/block_bins)); chosen=rng.choice(starts,size=blocks,replace=True)
            index=(chosen[:,None]+np.arange(block_bins)[None,:]).reshape(-1)[:len(sums)]
            draw_sum+=sums[index].sum(); draw_count+=counts[index].sum()
        draws.append(draw_sum/draw_count)
    return {"estimate":float(total_sum/total_count),"lower":float(np.quantile(draws,.025)),"upper":float(np.quantile(draws,.975)),"resamples":resamples,"block_bins":block_bins,"strata":len(strata),"seed":seed}


def summarize_real_transfer(farcal_root,online_root,prediction_root,output_path,dataset="pems_bay",model_seeds=(11,22,33),fault_seeds=(101,202,303),resamples=2000,scenario="C3",artifact_prefix=""):
    differences=defaultdict(lambda:defaultdict(list)); scores=[]; coverages=[]; by_run={}; sources=[]
    for fault_seed in fault_seeds:
      for model_seed in model_seeds:
        run_name=f"{artifact_prefix}{dataset}-seed{model_seed}"
        far_dir=Path(farcal_root)/f"{run_name}-{scenario}-fault{fault_seed}-far_cal_asymmetric-asym100"
        online_dir=Path(online_root)/f"{run_name}-{scenario}-feedback-{scenario}-fault{fault_seed}"
        test_dir=Path(prediction_root)/run_name/f"{scenario}-fault{fault_seed}"
        far,far_manifest=_cache_manifest(far_dir); online,online_manifest=_cache_manifest(online_dir); test,test_manifest=_cache_manifest(test_dir)
        fa=_cache_arrays(far_dir,far); oa=_align_origins(_cache_arrays(online_dir,online),fa["issue_bin"]); ta=_align_origins(_cache_arrays(test_dir,test),fa["issue_bin"])
        mask=ta["original_valid"][:,np.asarray(far["report_horizons"])-1]&fa["evaluation_mask"][:,None,:]
        truth=ta["truth_mph"][:,np.asarray(far["report_horizons"])-1]
        fi=far["methods"].index("far_cal_asymmetric")
        far_parts=[_interval_arrays(truth[:,hi],fa["lower_mph"][:,fi,0,hi],fa["upper_mph"][:,fi,0,hi],mask[:,hi],.1) for hi in range(len(far["report_horizons"]))]
        far_score=np.stack([x[0] for x in far_parts],axis=1); far_coverage=np.stack([x[1] for x in far_parts],axis=1)
        scores.append(far_score); coverages.append(far_coverage); comparisons={}
        for method in ("rolling","aci"):
            oi=online["methods"].index(method)
            comparator=np.stack([_interval_arrays(truth[:,hi],oa["lower_mph"][:,oi,0,hi],oa["upper_mph"][:,oi,0,hi],mask[:,hi],.1)[0] for hi in range(len(far["report_horizons"]))],axis=1)
            difference=comparator-far_score; differences[method][fault_seed].append(difference)
            comparisons[method+"_minus_far_cal"]=moving_block_mean_ci(difference,24,resamples,0)
        by_run[f"model{model_seed}-fault{fault_seed}"]={"far_cal_interval_score":float(np.nanmean(far_score)),"far_cal_coverage":float(np.nanmean(far_coverage)),"comparators":comparisons}
        sources.append({"model_seed":model_seed,"fault_seed":fault_seed,"farcal_manifest_sha256":sha256_file(far_manifest),"online_manifest_sha256":sha256_file(online_manifest),"test_manifest_sha256":sha256_file(test_manifest)})
    primary={}
    for method,by_fault in differences.items():
        strata=[]
        for values in by_fault.values():
            stacked=np.stack(values); count=np.isfinite(stacked).sum(axis=0)
            strata.append(np.divide(np.nansum(stacked,axis=0),count,out=np.full(count.shape,np.nan),where=count>0))
        primary[method+"_minus_far_cal"]=_stratified_block_ci(strata,24,resamples,0)
    score=float(np.nanmean(np.stack(scores))); coverage=float(np.nanmean(np.stack(coverages))); superiority=coverage>=.88 and all(x["lower"]>0 for x in primary.values())
    result={"schema_version":1,"status":"completed","dataset":dataset,"backbone":artifact_prefix.rstrip("-") or "far-gw","scenario":scenario,"candidate_id":"asym100","variant":"scalable_local_hourly_fault_focus",
        "model_seeds":list(model_seeds),"fault_seeds":list(fault_seeds),"evaluation_stride_bins":12,"focus_recovery_bins":12,"bootstrap_unit":"24 selected hourly origins, stratified by fault seed",
        "bootstrap_resamples":resamples,"difference_direction":"positive means the comparator has worse 90% equal-horizon interval score than FAR-Cal",
        "far_cal_interval_score":score,"far_cal_coverage":coverage,"coverage_floor":.88,"primary_comparisons":primary,"superiority_rule_met":superiority,"by_run":by_run,"sources":sources}
    write_json(output_path,result); return result


def summarize_sumo_farcal_ablation(farcal_root,online_root,prediction_root,output_path,model_seeds=(11,22,33),fault_seed=101,resamples=2000):
    by_seed={}; effects=defaultdict(list); sources=[]
    for model_seed in model_seeds:
        far_dir=Path(farcal_root)/f"sumo-seed{model_seed}-C3-fault{fault_seed}-allmethods-subset12"
        online_dir=Path(online_root)/f"sumo-seed{model_seed}-C3-feedback-C3-fault{fault_seed}"
        test_dir=Path(prediction_root)/f"sumo-seed{model_seed}"/f"C3-fault{fault_seed}"
        far,far_manifest=_cache_manifest(far_dir); online,online_manifest=_cache_manifest(online_dir); test,test_manifest=_cache_manifest(test_dir)
        fa=_cache_arrays(far_dir,far); oa=_align_origins(_cache_arrays(online_dir,online),fa["issue_bin"]); ta=_align_origins(_cache_arrays(test_dir,test),fa["issue_bin"])
        episodes=ta["episode_index"]; unique=np.unique(episodes); truth=ta["truth_mph"]; valid=ta["original_valid"]
        def primary(lower,upper):
            parts=[]
            for hi,horizon in enumerate(far["report_horizons"]):
                score=_interval_arrays(truth[:,horizon-1],lower[:,hi],upper[:,hi],valid[:,horizon-1],.1)[0]
                parts.append(score)
            values=np.stack(parts,axis=1)
            return np.asarray([np.nanmean(values[episodes==episode]) for episode in unique])
        method_values={method:primary(fa["lower_mph"][:,index,0],fa["upper_mph"][:,index,0]) for index,method in enumerate(far["methods"])}
        far_values=method_values["far_cal"]; seed_result={"far_cal_estimate":float(far_values.mean()),"ablations":{},"comparators":{}}
        for method in far["methods"]:
            if method=="far_cal": continue
            comparison=episode_mean_ci(method_values[method]-far_values,resamples,0); seed_result["ablations"][method]=comparison; effects[method].append(comparison["estimate"])
        for method in ("rolling","aci"):
            index=online["methods"].index(method)
            values=primary(oa["lower_mph"][:,index,0],oa["upper_mph"][:,index,0])
            comparison=episode_mean_ci(values-far_values,resamples,0); seed_result["comparators"][method+"_minus_far_cal"]=comparison; effects[method+"_minus_far_cal"].append(comparison["estimate"])
        by_seed[str(model_seed)]=seed_result
        sources.append({"model_seed":model_seed,"farcal_manifest_sha256":sha256_file(far_manifest),"online_manifest_sha256":sha256_file(online_manifest),"test_manifest_sha256":sha256_file(test_manifest)})
    result={"schema_version":1,"status":"completed","dataset":"sumo","scenario":"C3","fault_seed":fault_seed,
        "episode_subset":"configs/sumo_mechanism_subset12.json","model_seeds":list(model_seeds),"bootstrap_unit":"episode","bootstrap_resamples":resamples,
        "difference_direction":"positive means the ablation or comparator has worse 90% equal-horizon interval score than FAR-Cal",
        "by_model_seed":by_seed,"between_seed_mean_effect":{key:float(np.mean(values)) for key,values in effects.items()},"sources":sources}
    write_json(output_path,result); return result


def summarize_locked_sumo_farcal(farcal_root,online_root,prediction_root,output_path,candidate_id="n2b005w2",
        model_seeds=(11,22,33),fault_seeds=(101,202,303),resamples=2000):
    """Compare the tuning-locked FAR-Cal candidate with rolling and ACI on all held-out C3 runs."""
    by_run={}; effects=defaultdict(list); sources=[]
    for model_seed in model_seeds:
        for fault_seed in fault_seeds:
            far_dir=Path(farcal_root)/f"sumo-seed{model_seed}-C3-fault{fault_seed}-far_cal-{candidate_id}"
            online_dir=Path(online_root)/f"sumo-seed{model_seed}-C3-feedback-C3-fault{fault_seed}"
            test_dir=Path(prediction_root)/f"sumo-seed{model_seed}"/f"C3-fault{fault_seed}"
            far,far_manifest=_cache_manifest(far_dir); online,online_manifest=_cache_manifest(online_dir); test,test_manifest=_cache_manifest(test_dir)
            if far.get("candidate_id")!=candidate_id or far.get("episode_subset"):
                raise ValueError("locked FAR-Cal artifact does not match the requested full-test candidate")
            fa=_cache_arrays(far_dir,far); oa=_align_origins(_cache_arrays(online_dir,online),fa["issue_bin"]); ta=_align_origins(_cache_arrays(test_dir,test),fa["issue_bin"])
            episodes=ta["episode_index"]; unique=np.unique(episodes); truth=ta["truth_mph"]; valid=ta["original_valid"]
            def primary(lower,upper):
                values=np.stack([_interval_arrays(truth[:,horizon-1],lower[:,hi],upper[:,hi],valid[:,horizon-1],.1)[0]
                    for hi,horizon in enumerate(far["report_horizons"])],axis=1)
                return np.asarray([np.nanmean(values[episodes==episode]) for episode in unique])
            far_values=primary(fa["lower_mph"][:,far["methods"].index("far_cal"),0],fa["upper_mph"][:,far["methods"].index("far_cal"),0])
            comparisons={}
            for method in ("rolling","aci"):
                index=online["methods"].index(method)
                values=primary(oa["lower_mph"][:,index,0],oa["upper_mph"][:,index,0])
                comparison=episode_mean_ci(values-far_values,resamples,0)
                comparisons[method+"_minus_far_cal"]=comparison; effects[method].append(comparison["estimate"])
            key=f"model{model_seed}-fault{fault_seed}"
            by_run[key]={"far_cal_estimate":float(far_values.mean()),"comparators":comparisons}
            sources.append({"model_seed":model_seed,"fault_seed":fault_seed,"farcal_manifest_sha256":sha256_file(far_manifest),
                "online_manifest_sha256":sha256_file(online_manifest),"test_manifest_sha256":sha256_file(test_manifest)})
    mean_effect={method+"_minus_far_cal":float(np.mean(values)) for method,values in effects.items()}
    superior_runs={method+"_minus_far_cal":sum(by_run[key]["comparators"][method+"_minus_far_cal"]["lower"]>0 for key in by_run)
        for method in effects}
    result={"schema_version":1,"status":"completed","dataset":"sumo","scenario":"C3","candidate_id":candidate_id,
        "model_seeds":list(model_seeds),"fault_seeds":list(fault_seeds),"episode_count_per_run":30,"bootstrap_unit":"episode",
        "bootstrap_resamples":resamples,"difference_direction":"positive means the comparator has worse 90% equal-horizon interval score than FAR-Cal",
        "by_run":by_run,"mean_effect_across_runs":mean_effect,"runs_with_ci_above_zero":superior_runs,"sources":sources}
    write_json(output_path,result); return result


def summarize_sumo_replication(farcal_root,online_root,prediction_root,output_path,candidate_id="v2s25",
        model_seeds=(11,22,33),fault_seeds=(101,202,303),resamples=2000,partition="replication",method="far_cal",scenario="C3"):
    """Registered episode-clustered analysis of the untouched independent replication."""
    effects=defaultdict(list); candidate_scores=[]; candidate_coverages=[]; by_run={}; sources=[]; episode_order=None
    for model_seed in model_seeds:
        for fault_seed in fault_seeds:
            partition_suffix="" if partition=="test" else f"-{partition}"
            cache_name=f"{scenario}-fault{fault_seed}" if partition=="test" else f"{partition}-{scenario}-fault{fault_seed}"
            far_dir=Path(farcal_root)/f"sumo-seed{model_seed}-{scenario}-fault{fault_seed}-{method}{partition_suffix}-{candidate_id}"
            online_dir=Path(online_root)/f"sumo-seed{model_seed}-{scenario}-feedback-{scenario}-fault{fault_seed}{partition_suffix}"
            test_dir=Path(prediction_root)/f"sumo-seed{model_seed}"/cache_name
            far,far_manifest=_cache_manifest(far_dir); online,online_manifest=_cache_manifest(online_dir); test,test_manifest=_cache_manifest(test_dir)
            fa=_cache_arrays(far_dir,far); oa=_align_origins(_cache_arrays(online_dir,online),fa["issue_bin"]); ta=_align_origins(_cache_arrays(test_dir,test),fa["issue_bin"])
            episodes=ta["episode_index"]; unique=np.unique(episodes); truth=ta["truth_mph"]; valid=ta["original_valid"]
            if episode_order is None: episode_order=unique
            elif not np.array_equal(episode_order,unique): raise ValueError("replication episode order differs across runs")
            def primary(lower,upper):
                arrays=[_interval_arrays(truth[:,horizon-1],lower[:,hi],upper[:,hi],valid[:,horizon-1],.1) for hi,horizon in enumerate(far["report_horizons"])]
                score=np.stack([row[0] for row in arrays],axis=1); coverage=np.stack([row[1] for row in arrays],axis=1)
                return (np.asarray([np.nanmean(score[episodes==episode]) for episode in unique]),
                    np.asarray([np.nanmean(coverage[episodes==episode]) for episode in unique]))
            index=far["methods"].index(method); far_score,far_coverage=primary(fa["lower_mph"][:,index,0],fa["upper_mph"][:,index,0])
            candidate_scores.append(far_score); candidate_coverages.append(far_coverage); comparisons={}
            for comparator_method in ("rolling","aci"):
                oi=online["methods"].index(comparator_method); comparator,_=primary(oa["lower_mph"][:,oi,0],oa["upper_mph"][:,oi,0])
                difference=comparator-far_score; effects[comparator_method].append(difference); comparisons[comparator_method+"_minus_far_cal"]=episode_mean_ci(difference,resamples,0)
            key=f"model{model_seed}-fault{fault_seed}"; by_run[key]={"far_cal_estimate":float(far_score.mean()),"far_cal_coverage":float(far_coverage.mean()),"comparators":comparisons}
            sources.append({"model_seed":model_seed,"fault_seed":fault_seed,"farcal_manifest_sha256":sha256_file(far_manifest),
                "online_manifest_sha256":sha256_file(online_manifest),"test_manifest_sha256":sha256_file(test_manifest)})
    primary={method+"_minus_far_cal":episode_mean_ci(np.stack(values).mean(axis=0),resamples,0) for method,values in effects.items()}
    score=float(np.stack(candidate_scores).mean()); coverage=float(np.stack(candidate_coverages).mean())
    superiority=coverage>=.88 and all(value["lower"]>0 for value in primary.values())
    result={"schema_version":1,"status":"completed","dataset":"sumo" if partition=="test" else f"sumo_{partition}","scenario":scenario,"candidate_id":candidate_id,"method":method,
        "model_seeds":list(model_seeds),"fault_seeds":list(fault_seeds),"episode_indices":episode_order.tolist(),"bootstrap_unit":"episode",
        "bootstrap_resamples":resamples,"seed_aggregation":"paired effects averaged across all model/fault seeds within each episode before bootstrap",
        "difference_direction":"positive means the comparator has worse 90% equal-horizon interval score than FAR-Cal",
        "far_cal_interval_score":score,"far_cal_coverage":coverage,"coverage_floor":.88,"primary_comparisons":primary,
        "superiority_rule_met":superiority,"by_run":by_run,"sources":sources}
    write_json(output_path,result); return result
