"""Scalable sequential comparison of rolling residual intervals and arrival-batch ACI."""
from bisect import bisect_left,insort
from collections import defaultdict
import heapq
from pathlib import Path
import json
import math

import numpy as np

from ps1.utils.artifacts import sha256_file,write_json
from ps1.evaluation.bootstrap import moving_block_mean_ci,episode_mean_ci


LEVELS=(.90,.95)


class SortedNodeResiduals:
    """Exact per-node sliding target-time windows with deterministic order statistics."""
    def __init__(self,horizons,nodes,buffer_bins=2016):
        self.horizons=tuple(horizons); self.nodes=nodes; self.buffer_bins=buffer_bins; self.serial=0
        self.sorted={h:[[] for _ in range(nodes)] for h in self.horizons}
        self.expiry={h:[[] for _ in range(nodes)] for h in self.horizons}

    def insert(self,horizon,target,sensors,scores):
        for sensor,score in zip(np.asarray(sensors,dtype=int),np.asarray(scores,dtype=float)):
            if not np.isfinite(score) or score<0: raise ValueError("invalid residual score")
            insort(self.sorted[horizon][sensor],float(score))
            heapq.heappush(self.expiry[horizon][sensor],(int(target),self.serial,float(score))); self.serial+=1

    def expire(self,now):
        cutoff=now-self.buffer_bins
        for horizon in self.horizons:
            for sensor in range(self.nodes):
                heap=self.expiry[horizon][sensor]; ordered=self.sorted[horizon][sensor]
                while heap and heap[0][0]<cutoff:
                    _,_,score=heapq.heappop(heap); index=bisect_left(ordered,score)
                    if index==len(ordered) or ordered[index]!=score: raise RuntimeError("residual index inconsistency")
                    ordered.pop(index)

    def quantile(self,horizon,probability,conformal=False):
        output=np.empty(self.nodes,dtype=float); global_values=None
        for sensor,ordered in enumerate(self.sorted[horizon]):
            source=ordered
            if len(source)<50:
                if global_values is None: global_values=sorted(value for node in self.sorted[horizon] for value in node)
                source=global_values
            if not source: raise ValueError("empty received residual store")
            if conformal:
                alpha=1-probability
                if alpha<=0: output[sensor]=float("inf"); continue
                if alpha>=1: output[sensor]=0.; continue
                rank=math.ceil((len(source)+1)*probability)
                output[sensor]=float("inf") if rank>len(source) else source[rank-1]
            else:
                output[sensor]=source[math.ceil((len(source)-1)*probability)]
        return output


def _cache_manifest(directory):
    path=Path(directory)/"manifest.json"; return json.loads(path.read_text(encoding="utf-8")),path


def _initialize_store(calibration_dir,horizons,nodes,now,buffer_bins,calibration_fault_dir=None):
    manifest,_=_cache_manifest(calibration_dir); store=SortedNodeResiduals(horizons,nodes,buffer_bins)
    cutoff=now-buffer_bins
    calibration_arrival=None
    if calibration_fault_dir is not None:
        calibration_fault_dir=Path(calibration_fault_dir)
        cal_fault=json.loads((calibration_fault_dir/"fault_manifest.json").read_text(encoding="utf-8"))
        with np.load(calibration_fault_dir/"release_schedule.npz") as schedule:
            calibration_arrival=schedule["feedback_arrival"].copy()
        calibration_start=int(cal_fault["partition_start_row"])
    for chunk in manifest["chunks"]:
        path=Path(calibration_dir)/chunk["file"]
        if sha256_file(path)!=chunk["sha256"]: raise ValueError("calibration cache hash mismatch")
        with np.load(path) as data:
            scale=np.maximum((data["q95_mph"]-data["q05_mph"])/2,1.)
            score=np.abs(data["truth_mph"]-data["q50_mph"])/scale; valid=data["original_valid"]
            for row,issue in enumerate(data["issue_bin"]):
                for horizon in horizons:
                    target=int(issue)+horizon
                    if target<cutoff or target>now: continue
                    mask=valid[row,horizon-1].copy()
                    if calibration_arrival is not None:
                        local=target-calibration_start
                        mask &= (calibration_arrival[local]>=0)&(calibration_arrival[local]<=now-calibration_start)
                    sensors=np.flatnonzero(mask)
                    store.insert(horizon,target,sensors,score[row,horizon-1,sensors])
    store.expire(now); return store


def _arrival_order(feedback_arrival,max_now):
    observed,sensors=np.nonzero((feedback_arrival>=0)&(feedback_arrival<=max_now))
    arrivals=feedback_arrival[observed,sensors]; order=np.argsort(arrivals,kind="stable")
    return arrivals[order],observed[order],sensors[order]


class OnlineMetricSums:
    def __init__(self): self.count=0; self.covered=0; self.width=0.; self.score=0.; self.unbounded=0
    def add(self,truth,lower,upper,valid,alpha):
        y=truth[valid]; lo=lower[valid]; hi=upper[valid]; width=hi-lo
        self.count+=len(y); self.covered+=int(((lo<=y)&(y<=hi)).sum()); self.unbounded+=int((~np.isfinite(width)).sum())
        if np.isfinite(width).all():
            self.width+=float(width.sum()); self.score+=float((width+np.where(y<lo,2/alpha*(lo-y),0)+np.where(y>hi,2/alpha*(y-hi),0)).sum())
        else: self.width=float("inf"); self.score=float("inf")
    def result(self):
        finite=np.isfinite(self.width)
        return {"count":self.count,"picp":self.covered/self.count,"mpiw_mph":self.width/self.count if finite else None,
                "mean_interval_score":self.score/self.count if finite else None,"unbounded_count":self.unbounded}


def _cache_arrays(directory,manifest):
    collected=defaultdict(list)
    for chunk in manifest["chunks"]:
        path=Path(directory)/chunk["file"]
        if sha256_file(path)!=chunk["sha256"]: raise ValueError("cache chunk hash mismatch")
        with np.load(path) as data:
            for key in data.files: collected[key].append(data[key].copy())
    return {key:np.concatenate(parts) for key,parts in collected.items()}


def _episodic_store(calibration_dir,calibration,horizons,nodes,first_local,buffer_bins,bins_per_episode,episode_subset=None):
    arrays=_cache_arrays(calibration_dir,calibration); episodes=arrays["episode_index"]
    if episode_subset is not None:
        keep=np.isin(episodes,episode_subset); arrays={key:value[keep] if isinstance(value,np.ndarray) and len(value)==len(episodes) else value for key,value in arrays.items()}
        episodes=arrays["episode_index"]
    unique=np.unique(episodes); rank={int(value):index for index,value in enumerate(unique)}
    test_base=len(unique)*288; now=test_base+first_local
    store=SortedNodeResiduals(horizons,nodes,buffer_bins); cutoff=now-buffer_bins
    scale=np.maximum((arrays["q95_mph"]-arrays["q05_mph"])/2,1.)
    score=np.abs(arrays["truth_mph"]-arrays["q50_mph"])/scale; valid=arrays["original_valid"]
    for row,issue in enumerate(arrays["issue_bin"]):
        episode=int(episodes[row]); local=int(issue)-episode*bins_per_episode
        mapped_issue=rank[episode]*288+local
        for horizon in horizons:
            target=mapped_issue+horizon
            if target<cutoff or target>now: continue
            mask=valid[row,horizon-1]; sensors=np.flatnonzero(mask)
            store.insert(horizon,target,sensors,score[row,horizon-1,sensors])
    store.expire(now); return store,test_base


def _evaluate_received_online_episodic(calibration_dir,test_cache_dir,calibration,test,
        calibration_manifest,test_manifest,point_fault_dir,feedback_fault_dir,point_manifest,
        feedback_manifest,point_manifest_path,feedback_manifest_path,feedback_arrival,output,
        report_horizons,buffer_bins,aci_learning_rates,episode_subset,calibration_episode_subset):
    arrays=_cache_arrays(test_cache_dir,test); episodes=arrays["episode_index"]
    all_test_episode_indices=np.asarray(point_manifest["test_episode_indices"],dtype=int)
    bins_per_episode=point_manifest["test_length"]//len(all_test_episode_indices)
    test_episode_indices=all_test_episode_indices if episode_subset is None else np.asarray(episode_subset,dtype=int)
    if len(np.unique(test_episode_indices))!=len(test_episode_indices) or not np.isin(test_episode_indices,all_test_episode_indices).all():
        raise ValueError("episode subset must contain unique registered test episode indices")
    full_ordinal={int(episode):index for index,episode in enumerate(all_test_episode_indices)}
    nodes=test["sensors"]; rates=aci_learning_rates or {.90:.005,.95:.005}
    accumulators={(method,level,h):OnlineMetricSums() for method in ("rolling","aci") for level in LEVELS for h in report_horizons}
    chunks=[]; output.mkdir(parents=True,exist_ok=True)
    for ordinal,episode in enumerate(test_episode_indices):
        rows=np.flatnonzero(episodes==episode)
        if not len(rows): raise ValueError(f"test cache omits episode {episode}")
        source_ordinal=full_ordinal[int(episode)]
        issue_bins=arrays["issue_bin"][rows]; local_issues=issue_bins-(point_manifest["test_start_row"]+source_ordinal*bins_per_episode)
        first_local=int(local_issues[0])
        store,test_base=_episodic_store(calibration_dir,calibration,report_horizons,nodes,first_local,buffer_bins,bins_per_episode,calibration_episode_subset)
        control={(h,l):1-l for h in report_horizons for l in LEVELS}; pending={}
        start=source_ordinal*bins_per_episode; stop=start+bins_per_episode
        local_feedback=np.where(feedback_arrival[start:stop]>=0,feedback_arrival[start:stop]-start,-1)
        arrival,observed_local,arrival_sensor=_arrival_order(local_feedback,int(local_issues[-1])); arrival_pointer=0
        lower_out=np.empty((len(rows),2,2,len(report_horizons),nodes),dtype=np.float32); upper_out=np.empty_like(lower_out)
        point=arrays["q50_mph"][rows]; scale=np.maximum((arrays["q95_mph"][rows]-arrays["q05_mph"][rows])/2,1.)
        truth=arrays["truth_mph"][rows]; valid=arrays["original_valid"][rows]
        for row,local_issue in enumerate(local_issues):
            local_issue=int(local_issue); now=test_base+local_issue; batch_misses=defaultdict(list)
            end=np.searchsorted(arrival,local_issue,side="right")
            for index in range(arrival_pointer,end):
                target=int(observed_local[index]); sensor=int(arrival_sensor[index])
                for horizon in report_horizons:
                    record=pending.get((target,horizon))
                    if record is None or not record["valid"][sensor]: continue
                    score=abs(record["truth"][sensor]-record["point"][sensor])/record["scale"][sensor]
                    store.insert(horizon,test_base+target,[sensor],[score])
                    for level_index,level in enumerate(LEVELS):
                        batch_misses[(horizon,level)].append(float(not record["lower"][level_index,sensor]<=record["truth"][sensor]<=record["upper"][level_index,sensor]))
            arrival_pointer=end
            for key,misses in batch_misses.items():
                horizon,level=key; control[key]+=rates[level]*((1-level)-float(np.mean(misses)))
            store.expire(now)
            for horizon_index,horizon in enumerate(report_horizons):
                model_index=horizon-1; p=point[row,model_index]; s=scale[row,model_index]; y=truth[row,model_index]; mask=valid[row,model_index]
                aci_bounds=np.empty((2,2,nodes),dtype=np.float32)
                for level_index,level in enumerate(LEVELS):
                    rolling_q=store.quantile(horizon,level); width=s*rolling_q
                    lower=np.maximum(0,p-width); upper=np.maximum(lower,p+width)
                    alpha=control[(horizon,level)]; aci_q=store.quantile(horizon,1-alpha,conformal=True); aci_width=s*aci_q
                    aci_lower=np.maximum(0,p-aci_width); aci_upper=np.maximum(aci_lower,p+aci_width)
                    lower_out[row,0,level_index,horizon_index]=lower; upper_out[row,0,level_index,horizon_index]=upper
                    lower_out[row,1,level_index,horizon_index]=aci_lower; upper_out[row,1,level_index,horizon_index]=aci_upper
                    aci_bounds[level_index,0]=aci_lower; aci_bounds[level_index,1]=aci_upper
                    accumulators[("rolling",level,horizon)].add(y,lower,upper,mask,1-level)
                    accumulators[("aci",level,horizon)].add(y,aci_lower,aci_upper,mask,1-level)
                pending[(local_issue+horizon,horizon)]={"point":p.copy(),"scale":s.copy(),"truth":y.copy(),"valid":mask.copy(),
                    "lower":aci_bounds[:,0].copy(),"upper":aci_bounds[:,1].copy()}
        output_path=output/f"episode-{ordinal:02d}.npz"
        np.savez_compressed(output_path,issue_bin=issue_bins,episode_index=np.full(len(rows),episode,dtype=np.int16),lower_mph=lower_out,upper_mph=upper_out)
        chunks.append({"file":output_path.name,"sha256":sha256_file(output_path),"origins":len(rows),"episode_index":int(episode)})
    metrics={method:{str(level):{str(h):accumulators[(method,level,h)].result() for h in report_horizons} for level in LEVELS} for method in ("rolling","aci")}
    result={"schema_version":1,"status":"completed","dataset":test["dataset"],"model_seed":test["checkpoint_training_seed"],
        "point_scenario":point_manifest["scenario_id"],"point_fault_seed":point_manifest["seed"],
        "feedback_scenario":feedback_manifest["scenario_id"],"feedback_fault_seed":feedback_manifest["seed"],
        "episode_reset":True,"episode_count":len(test_episode_indices),"episode_indices":test_episode_indices.tolist(),
        "episode_subset":episode_subset is not None,"calibration_episode_indices":None if calibration_episode_subset is None else list(calibration_episode_subset),"bins_per_episode":bins_per_episode,
        "report_horizons":list(report_horizons),"levels":list(LEVELS),"methods":["rolling","aci"],"metrics":metrics,
        "aci_learning_rates":{str(k):v for k,v in rates.items()},"guarantee":"ACI is a spatial arrival-batch project variant; no inherited finite-sample claim",
        "calibration_manifest_sha256":sha256_file(calibration_manifest),"test_manifest_sha256":sha256_file(test_manifest),
        "point_fault_manifest_sha256":sha256_file(point_manifest_path),"feedback_fault_manifest_sha256":sha256_file(feedback_manifest_path),
        "chunks":chunks,"chunk_arrays":{"lower_mph/upper_mph":"float32 [origin,method,level,horizon,sensor]","method_order":["rolling","aci"]}}
    write_json(output/"manifest.json",result); return result


def evaluate_received_online(calibration_dir,test_cache_dir,point_fault_dir,feedback_fault_dir,output_dir,
                             report_horizons=(3,6,12),buffer_bins=2016,aci_learning_rates=None,episode_subset=None,
                             calibration_episode_subset=None,calibration_fault_dir=None):
    """Compare rolling and ACI with fixed cached predictions under one feedback schedule."""
    calibration,calibration_manifest=_cache_manifest(calibration_dir); test,test_manifest=_cache_manifest(test_cache_dir)
    point_fault_dir,feedback_fault_dir,output=map(Path,(point_fault_dir,feedback_fault_dir,output_dir))
    point_manifest_path=point_fault_dir/"fault_manifest.json"; feedback_manifest_path=feedback_fault_dir/"fault_manifest.json"
    point_manifest=json.loads(point_manifest_path.read_text(encoding="utf-8")); feedback_manifest=json.loads(feedback_manifest_path.read_text(encoding="utf-8"))
    with np.load(point_fault_dir/"release_schedule.npz") as point_schedule,np.load(feedback_fault_dir/"release_schedule.npz") as feedback_schedule:
        if not np.array_equal(point_schedule["input_arrival"],feedback_schedule["input_arrival"]): raise ValueError("input schedules differ; point forecasts would not be shared")
        feedback_arrival=feedback_schedule["feedback_arrival"].copy()
    if test["scenario"]!=point_manifest["scenario_id"] or test["release_schedule_sha256"]!=sha256_file(point_fault_dir/"release_schedule.npz"):
        raise ValueError("test cache does not match point-input schedule")
    if calibration["checkpoint_sha256"]!=test["checkpoint_sha256"]: raise ValueError("calibration and test checkpoints differ")
    if test.get("episode_aware"):
        if calibration_fault_dir is not None: raise ValueError("cold archive is real-data only")
        return _evaluate_received_online_episodic(
            calibration_dir,test_cache_dir,calibration,test,calibration_manifest,test_manifest,
            point_fault_dir,feedback_fault_dir,point_manifest,feedback_manifest,point_manifest_path,
            feedback_manifest_path,feedback_arrival,output,report_horizons,buffer_bins,aci_learning_rates,episode_subset,calibration_episode_subset)
    first_issue=test["chunks"][0]["first_issue_bin"]; test_start=int(point_manifest["test_start_row"]); nodes=test["sensors"]
    store=_initialize_store(calibration_dir,report_horizons,nodes,first_issue,buffer_bins,calibration_fault_dir)
    rates=aci_learning_rates or {.90:.005,.95:.005}; control={(h,l):1-l for h in report_horizons for l in LEVELS}
    accumulators={(method,level,h):OnlineMetricSums() for method in ("rolling","aci") for level in LEVELS for h in report_horizons}
    max_now=test["chunks"][-1]["last_issue_bin"]-test_start
    arrival,observed_local,arrival_sensor=_arrival_order(feedback_arrival,max_now); arrival_pointer=0
    pending={}; chunks=[]; output.mkdir(parents=True,exist_ok=True)
    for chunk_meta in test["chunks"]:
        path=Path(test_cache_dir)/chunk_meta["file"]
        if sha256_file(path)!=chunk_meta["sha256"]: raise ValueError("test cache hash mismatch")
        with np.load(path) as data:
            issue_bins=data["issue_bin"]; point=data["q50_mph"]; scale=np.maximum((data["q95_mph"]-data["q05_mph"])/2,1.)
            truth=data["truth_mph"]; valid=data["original_valid"]
            lower_out=np.empty((len(issue_bins),2,2,len(report_horizons),nodes),dtype=np.float32)
            upper_out=np.empty_like(lower_out)
            for row,issue_value in enumerate(issue_bins):
                issue=int(issue_value); now=issue-test_start; batch_misses=defaultdict(list)
                end=np.searchsorted(arrival,now,side="right")
                for index in range(arrival_pointer,end):
                    target=test_start+int(observed_local[index]); sensor=int(arrival_sensor[index])
                    for horizon in report_horizons:
                        record=pending.get((target,horizon))
                        if record is None or not record["valid"][sensor]: continue
                        score=abs(record["truth"][sensor]-record["point"][sensor])/record["scale"][sensor]
                        store.insert(horizon,target,[sensor],[score])
                        for level_index,level in enumerate(LEVELS):
                            batch_misses[(horizon,level)].append(float(not record["lower"][level_index,sensor]<=record["truth"][sensor]<=record["upper"][level_index,sensor]))
                arrival_pointer=end
                for key,misses in batch_misses.items():
                    horizon,level=key; control[key]+=rates[level]*((1-level)-float(np.mean(misses)))
                store.expire(issue)
                for horizon_index,horizon in enumerate(report_horizons):
                    model_index=horizon-1; p=point[row,model_index]; s=scale[row,model_index]; y=truth[row,model_index]; mask=valid[row,model_index]
                    aci_bounds=np.empty((2,2,nodes),dtype=np.float32)
                    for level_index,level in enumerate(LEVELS):
                        rolling_q=store.quantile(horizon,level); rolling_width=s*rolling_q
                        rolling_lower=np.maximum(0,p-rolling_width); rolling_upper=np.maximum(rolling_lower,p+rolling_width)
                        alpha=control[(horizon,level)]; aci_q=store.quantile(horizon,1-alpha,conformal=True); aci_width=s*aci_q
                        aci_lower=np.maximum(0,p-aci_width); aci_upper=np.maximum(aci_lower,p+aci_width)
                        lower_out[row,0,level_index,horizon_index]=rolling_lower; upper_out[row,0,level_index,horizon_index]=rolling_upper
                        lower_out[row,1,level_index,horizon_index]=aci_lower; upper_out[row,1,level_index,horizon_index]=aci_upper
                        aci_bounds[level_index,0]=aci_lower; aci_bounds[level_index,1]=aci_upper
                        accumulators[("rolling",level,horizon)].add(y,rolling_lower,rolling_upper,mask,1-level)
                        accumulators[("aci",level,horizon)].add(y,aci_lower,aci_upper,mask,1-level)
                    pending[(issue+horizon,horizon)]={"point":p.copy(),"scale":s.copy(),"truth":y.copy(),"valid":mask.copy(),
                                                       "lower":aci_bounds[:,0].copy(),"upper":aci_bounds[:,1].copy()}
                expired=[key for key in pending if key[0]+buffer_bins<issue]
                for key in expired: del pending[key]
        output_path=output/f"chunk-{len(chunks):04d}.npz"; np.savez_compressed(output_path,issue_bin=issue_bins,lower_mph=lower_out,upper_mph=upper_out)
        chunks.append({"file":output_path.name,"sha256":sha256_file(output_path),"origins":len(issue_bins)})
    metrics={method:{str(level):{str(h):accumulators[(method,level,h)].result() for h in report_horizons} for level in LEVELS} for method in ("rolling","aci")}
    result={"schema_version":1,"status":"completed","dataset":test["dataset"],"model_seed":test["checkpoint_training_seed"],
            "point_scenario":point_manifest["scenario_id"],"point_fault_seed":point_manifest["seed"],
            "feedback_scenario":feedback_manifest["scenario_id"],"feedback_fault_seed":feedback_manifest["seed"],
            "report_horizons":list(report_horizons),"levels":list(LEVELS),"methods":["rolling","aci"],"metrics":metrics,
            "aci_learning_rates":{str(k):v for k,v in rates.items()},"aci_final_control":{f"h{h}_l{level}":value for (h,level),value in control.items()},
            "guarantee":"ACI is a spatial arrival-batch project variant; no inherited finite-sample claim",
            "calibration_manifest_sha256":sha256_file(calibration_manifest),"test_manifest_sha256":sha256_file(test_manifest),
            "point_fault_manifest_sha256":sha256_file(point_manifest_path),"feedback_fault_manifest_sha256":sha256_file(feedback_manifest_path),
            "calibration_fault_manifest_sha256":sha256_file(Path(calibration_fault_dir)/"fault_manifest.json") if calibration_fault_dir is not None else None,
            "chunks":chunks,"chunk_arrays":{"lower_mph/upper_mph":"float32 [origin,method,level,horizon,sensor]","method_order":["rolling","aci"]}}
    write_json(output/"manifest.json",result); return result


def _interval_arrays(truth,lower,upper,valid,alpha):
    width=upper-lower
    score=width+np.where(truth<lower,2/alpha*(lower-truth),0)+np.where(truth>upper,2/alpha*(truth-upper),0)
    covered=((lower<=truth)&(truth<=upper)).astype(float)
    return tuple(np.where(valid,value,np.nan) for value in (score,covered,width))


def _align_origins(arrays,issue_bins):
    if np.array_equal(arrays["issue_bin"],issue_bins): return arrays
    full_issue=arrays["issue_bin"]; indices=np.searchsorted(full_issue,issue_bins)
    if (indices>=len(full_issue)).any() or not np.array_equal(full_issue[indices],issue_bins):
        raise ValueError("cache omits requested issue bins")
    return {key:(value[indices] if len(value)==len(full_issue) else value) for key,value in arrays.items()}


def compare_feedback_arms(immediate_dir,delayed_dir,test_cache_dir,output_path,resamples=2000,block_bins=288,seed=0):
    immediate,immediate_manifest=_cache_manifest(immediate_dir); delayed,delayed_manifest=_cache_manifest(delayed_dir); test,test_manifest=_cache_manifest(test_cache_dir)
    for key in ("dataset","model_seed","point_scenario","point_fault_seed","test_manifest_sha256"):
        if immediate.get(key)!=delayed.get(key): raise ValueError(f"feedback arms differ on {key}")
    if immediate["feedback_scenario"]==delayed["feedback_scenario"]: raise ValueError("feedback scenarios must differ")
    ia=_cache_arrays(immediate_dir,immediate); da=_cache_arrays(delayed_dir,delayed); ta=_cache_arrays(test_cache_dir,test)
    if not np.array_equal(ia["issue_bin"],da["issue_bin"]): raise ValueError("online issue bins differ")
    ta=_align_origins(ta,ia["issue_bin"])
    episodic=bool(test.get("episode_aware")); episodes=ta.get("episode_index")
    if episodic and episodes is None: raise ValueError("episode-aware cache omits episode_index")
    collected={}
    for method_index,method in enumerate(immediate["methods"]):
        for level_index,level in enumerate(immediate["levels"]):
            for horizon_index,horizon in enumerate(immediate["report_horizons"]):
                truth=ta["truth_mph"][:,horizon-1]; valid=ta["original_valid"][:,horizon-1]
                iarrays=_interval_arrays(truth,ia["lower_mph"][:,method_index,level_index,horizon_index],ia["upper_mph"][:,method_index,level_index,horizon_index],valid,1-level)
                darrays=_interval_arrays(truth,da["lower_mph"][:,method_index,level_index,horizon_index],da["upper_mph"][:,method_index,level_index,horizon_index],valid,1-level)
                for metric_index,metric in enumerate(("interval_score","coverage","width")):
                    collected[(method,level,horizon,metric)]=darrays[metric_index]-iarrays[metric_index]
    def confidence_interval(values):
        if not episodic: return moving_block_mean_ci(values,block_bins,resamples,seed)
        per_episode=np.asarray([np.nanmean(values[episodes==episode]) for episode in np.unique(episodes)])
        return episode_mean_ci(per_episode,resamples,seed)
    comparisons={}
    for method in immediate["methods"]:
        comparisons[method]={}
        for level in immediate["levels"]:
            comparisons[method][str(level)]={}
            for horizon in immediate["report_horizons"]:
                metrics={}
                for metric in ("interval_score","coverage","width"):
                    values=collected[(method,level,horizon,metric)]
                    metrics[metric+"_delayed_minus_immediate"]=confidence_interval(values)
                comparisons[method][str(level)][str(horizon)]=metrics
        primary=np.stack([collected[(method,.90,h,"interval_score")] for h in immediate["report_horizons"]],axis=1)
        comparisons[method]["primary_equal_horizon_90_interval_score_delayed_minus_immediate"]=confidence_interval(primary)
    result={"schema_version":1,"status":"completed_single_seed_single_fault","dataset":immediate["dataset"],"model_seed":immediate["model_seed"],
            "point_scenario":immediate["point_scenario"],"fault_seed":immediate["point_fault_seed"],
            "immediate_feedback_scenario":immediate["feedback_scenario"],"delayed_feedback_scenario":delayed["feedback_scenario"],
            "paired_target_stream":True,"bootstrap_unit":"episode" if episodic else "time_block","block_bins":None if episodic else block_bins,"bootstrap_resamples":resamples,"bootstrap_seed":seed,
            "comparisons":comparisons,"interpretation":"Positive interval-score differences favor immediate feedback; confidence intervals crossing zero are inconclusive.",
            "limitations":["One model seed","One fault seed","Rolling and project-variant ACI only","H1 evidence incomplete until registered replication"],
            "immediate_manifest_sha256":sha256_file(immediate_manifest),"delayed_manifest_sha256":sha256_file(delayed_manifest),"test_manifest_sha256":sha256_file(test_manifest)}
    write_json(output_path,result); return result


def compare_input_mechanism(reference_dir,candidate_dir,reference_test_dir,candidate_test_dir,output_path,resamples=2000,seed=0):
    """Paired C2-minus-C0 comparison on a prespecified SUMO episode subset."""
    reference,reference_manifest=_cache_manifest(reference_dir); candidate,candidate_manifest=_cache_manifest(candidate_dir)
    reference_test,reference_test_manifest=_cache_manifest(reference_test_dir); candidate_test,candidate_test_manifest=_cache_manifest(candidate_test_dir)
    for key in ("dataset","model_seed","episode_count","episode_indices"):
        if reference.get(key)!=candidate.get(key): raise ValueError(f"mechanism arms differ on {key}")
    if reference["point_scenario"]!="C0" or candidate["point_scenario"]!="C2": raise ValueError("input mechanism comparison requires C0 reference and C2 candidate")
    ra=_cache_arrays(reference_dir,reference); ca=_cache_arrays(candidate_dir,candidate)
    if not np.array_equal(ra["issue_bin"],ca["issue_bin"]): raise ValueError("mechanism issue bins differ")
    rt=_align_origins(_cache_arrays(reference_test_dir,reference_test),ra["issue_bin"])
    ct=_align_origins(_cache_arrays(candidate_test_dir,candidate_test),ra["issue_bin"])
    for key in ("issue_bin","truth_mph","original_valid","episode_index"):
        if not np.array_equal(rt[key],ct[key]): raise ValueError(f"paired target streams differ on {key}")
    episodes=rt["episode_index"]; unique=np.unique(episodes)
    def ci(values):
        return episode_mean_ci(np.asarray([np.nanmean(values[episodes==episode]) for episode in unique]),resamples,seed)
    comparisons={"point":{},"intervals":{}}
    for horizon in reference["report_horizons"]:
        truth=rt["truth_mph"][:,horizon-1]; valid=rt["original_valid"][:,horizon-1]
        difference=np.where(valid,np.abs(ct["q50_mph"][:,horizon-1]-truth)-np.abs(rt["q50_mph"][:,horizon-1]-truth),np.nan)
        comparisons["point"][str(horizon)]={"mae_c2_minus_c0":ci(difference)}
    for method_index,method in enumerate(reference["methods"]):
        comparisons["intervals"][method]={}
        for level_index,level in enumerate(reference["levels"]):
            comparisons["intervals"][method][str(level)]={}
            for horizon_index,horizon in enumerate(reference["report_horizons"]):
                truth=rt["truth_mph"][:,horizon-1]; valid=rt["original_valid"][:,horizon-1]
                rvalues=_interval_arrays(truth,ra["lower_mph"][:,method_index,level_index,horizon_index],ra["upper_mph"][:,method_index,level_index,horizon_index],valid,1-level)
                cvalues=_interval_arrays(truth,ca["lower_mph"][:,method_index,level_index,horizon_index],ca["upper_mph"][:,method_index,level_index,horizon_index],valid,1-level)
                comparisons["intervals"][method][str(level)][str(horizon)]={
                    name+"_c2_minus_c0":ci(cvalues[index]-rvalues[index])
                    for index,name in enumerate(("interval_score","coverage","width"))}
    result={"schema_version":1,"status":"completed_input_mechanism_subset","dataset":reference["dataset"],
        "model_seed":reference["model_seed"],"fault_seed":candidate["point_fault_seed"],"reference":"C0","candidate":"C2",
        "episode_indices":reference["episode_indices"],"episode_count":reference["episode_count"],"bootstrap_unit":"episode",
        "bootstrap_resamples":resamples,"bootstrap_seed":seed,"comparisons":comparisons,
        "reference_online_manifest_sha256":sha256_file(reference_manifest),"candidate_online_manifest_sha256":sha256_file(candidate_manifest),
        "reference_test_manifest_sha256":sha256_file(reference_test_manifest),"candidate_test_manifest_sha256":sha256_file(candidate_test_manifest)}
    write_json(output_path,result); return result
