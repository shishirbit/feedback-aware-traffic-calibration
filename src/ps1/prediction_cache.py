"""Chunked prediction artifacts for frozen-backbone calibration comparisons."""
from dataclasses import fields
from pathlib import Path
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ps1.data.graphs import local_groups
from ps1.models.graph_wavenet_masked import GraphWaveNetConfig, MaskedGraphWaveNet
from ps1.models.staeformer_quantile import STAEformerQuantile, STAEformerQuantileConfig
from ps1.models.stid_quantile import STIDQuantile, STIDQuantileConfig
from ps1.models.dcrnn_quantile import DCRNNQuantile, DCRNNQuantileConfig
from ps1.training import TrafficWindowDataset, _row_normalize
from ps1.utils.artifacts import sha256_file, write_json


SCHEMA_VERSION = 1


def load_checkpoint_model(checkpoint_path, dataset, device="cpu"):
    # Checkpoints are produced locally by train_registered and verified by SHA-256
    # before registration. Seed 11 predates list-only metadata, so trusted-local
    # loading is required for its NumPy training-median array.
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    architecture = saved["architecture"]
    if architecture.get("version") == DCRNNQuantile.version:
        allowed = {item.name for item in fields(DCRNNQuantileConfig)}
        config = DCRNNQuantileConfig(**{key: value for key, value in architecture.items() if key in allowed})
        forward = torch.from_numpy(_row_normalize(dataset.adjacency)).to(device)
        reverse = torch.from_numpy(_row_normalize(dataset.adjacency.T)).to(device)
        model = DCRNNQuantile(config, (forward, reverse)).to(device)
    elif architecture.get("version") == STIDQuantile.version:
        allowed = {item.name for item in fields(STIDQuantileConfig)}
        config = STIDQuantileConfig(**{key: value for key, value in architecture.items() if key in allowed})
        model = STIDQuantile(config).to(device)
    elif architecture.get("version") == STAEformerQuantile.version:
        allowed = {item.name for item in fields(STAEformerQuantileConfig)}
        config = STAEformerQuantileConfig(**{key: value for key, value in architecture.items() if key in allowed})
        model = STAEformerQuantile(config).to(device)
    else:
        allowed = {item.name for item in fields(GraphWaveNetConfig)}
        config = GraphWaveNetConfig(**{key: value for key, value in architecture.items() if key in allowed})
        forward = torch.from_numpy(_row_normalize(dataset.adjacency)).to(device)
        reverse = torch.from_numpy(_row_normalize(dataset.adjacency.T)).to(device)
        model = MaskedGraphWaveNet(config, (forward, reverse)).to(device)
    model.load_state_dict(saved["model"]); model.eval()
    return model, saved


def context_statistics(dataset):
    """Fit context scaling on clean training histories only."""
    groups = local_groups(dataset.adjacency)
    width = max(map(len, groups)); indices = np.zeros((len(groups), width), dtype=np.int64)
    weights = np.zeros_like(indices, dtype=np.float64)
    for sensor, group in enumerate(groups):
        indices[sensor, :len(group)] = group; weights[sensor, :len(group)] = 1. / len(group)
    total = np.zeros(2); square = np.zeros(2); count = 0
    for issue in dataset.issues:
        rows = slice(int(issue)-dataset.history+1, int(issue)+1)
        available = (dataset.valid[rows][:, indices] * weights).sum(axis=2).mean(axis=0)
        mean_age = (dataset.age[rows][:, indices] * weights).sum(axis=2).mean(axis=0)
        values = np.stack((available, np.log1p(mean_age)), axis=1)
        total += values.sum(axis=0); square += np.square(values).sum(axis=0); count += len(values)
    mean = total/count; variance = np.maximum(square/count-np.square(mean), 1e-16)
    return mean, np.sqrt(variance), groups


def issue_contexts(dataset, issue_bins, groups, mean, std):
    contexts = np.empty((len(issue_bins), len(groups), 2), dtype=np.float32)
    for row, issue in enumerate(issue_bins):
        history = slice(int(issue)-dataset.history+1, int(issue)+1)
        for sensor, group in enumerate(groups):
            raw = np.array((dataset.valid[history][:, group].mean(),
                            np.log1p(dataset.age[history][:, group].mean())))
            contexts[row, sensor] = (raw-mean)/std
    return contexts


def _write_chunk(path, issue_bin, predictions, truth, valid, contexts, episode_index=None):
    low, median, high = predictions
    truth = np.where(valid, truth, np.nan).astype(np.float32)
    arrays={"issue_bin":np.asarray(issue_bin,dtype=np.int32),"q05_mph":low.astype(np.float32),
            "q50_mph":median.astype(np.float32),"q95_mph":high.astype(np.float32),"truth_mph":truth,
            "original_valid":np.asarray(valid,dtype=bool),"context":np.asarray(contexts,dtype=np.float32)}
    if episode_index is not None: arrays["episode_index"]=np.asarray(episode_index,dtype=np.int16)
    np.savez_compressed(path,**arrays)
    result={"file":path.name,"sha256":sha256_file(path),"origins":len(issue_bin),
            "first_issue_bin":int(issue_bin[0]),"last_issue_bin":int(issue_bin[-1])}
    if episode_index is not None:
        result.update({"first_episode_index":int(episode_index[0]),"last_episode_index":int(episode_index[-1])})
    return result


def faulted_history(dataset,input_arrival,test_start,issue):
    """Construct one history from releases visible at issue, including late packets."""
    issue=int(issue); test_start=int(test_start); local_issue=issue-test_start
    first=issue-dataset.history+1; nodes=dataset.values.shape[1]
    episode_start=0
    if dataset.episode_index is not None:
        episode=int(dataset.episode_index[issue]); episode_start=int(np.flatnonzero(dataset.episode_index==episode)[0])

    def received(rows):
        rows=np.asarray(rows,dtype=np.int64); result=np.zeros((len(rows),nodes),dtype=bool)
        before=rows<test_start
        if before.any(): result[before]=dataset.valid[rows[before]]
        after=~before
        if after.any():
            local=rows[after]-test_start; arrivals=input_arrival[local]
            result[after]=dataset.valid[rows[after]]&(arrivals>=0)&(arrivals<=local_issue)
        return result

    previous=np.full(nodes,first-1,dtype=np.int64); found=np.zeros(nodes,dtype=bool)
    while (previous>=episode_start).any() and not found.all():
        active=(previous>=episode_start)&~found; sensors=np.flatnonzero(active); rows=previous[sensors]
        before=rows<test_start; available=np.zeros(len(sensors),dtype=bool)
        if before.any(): available[before]=dataset.valid[rows[before],sensors[before]]
        after=~before
        if after.any():
            local=rows[after]-test_start; arrival=input_arrival[local,sensors[after]]
            available[after]=dataset.valid[rows[after],sensors[after]]&(arrival>=0)&(arrival<=local_issue)
        found[sensors[available]]=True; previous[sensors[~available]]-=1
    last_value=dataset.medians.copy(); last_bin=np.full(nodes,-1,dtype=np.int64)
    if found.any():
        sensors=np.flatnonzero(found); last_value[sensors]=dataset.values[previous[sensors],sensors]; last_bin[sensors]=previous[sensors]
    slots=np.arange(first,issue+1,dtype=np.int64); mask=received(slots)
    filled=np.empty((dataset.history,nodes),dtype=np.float32); age=np.empty_like(filled)
    for offset,slot in enumerate(slots):
        observed=mask[offset]; last_value[observed]=dataset.values[slot,observed]; last_bin[observed]=slot
        filled[offset]=last_value; age[offset]=np.where(last_bin>=episode_start,slot-last_bin,slot-episode_start+1)
    calendar=np.broadcast_to(dataset.calendar[slots,None,:],(dataset.history,nodes,4))
    features=np.concatenate((((filled-dataset.mean)/dataset.std)[...,None],mask[...,None].astype(np.float32),
                             np.minimum(age,288)[...,None]/288.,calendar),axis=-1).astype(np.float32)
    return features,mask,age


def cache_clean_predictions(prepared_path, checkpoint_path, output_dir, partition, batch_size=32,
                            chunk_origins=288, device="cpu"):
    if partition not in {"calibration", "test", "tune", "replication", "confirmation", "asym_validation"}:
        raise ValueError("unknown prediction cache partition")
    prepared_path, checkpoint_path, output = Path(prepared_path), Path(checkpoint_path), Path(output_dir)
    dataset = TrafficWindowDataset(prepared_path, partition, augment=False)
    train = TrafficWindowDataset(prepared_path, "train", augment=False)
    context_mean, context_std, groups = context_statistics(train)
    model, saved = load_checkpoint_model(checkpoint_path, dataset, device)
    input_fingerprint = {"prepared_sha256": sha256_file(prepared_path),
                         "checkpoint_sha256": sha256_file(checkpoint_path), "partition": partition}
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in input_fingerprint.items()):
            for chunk in existing["chunks"]:
                if sha256_file(output/chunk["file"]) != chunk["sha256"]:
                    raise ValueError("prediction cache chunk hash mismatch")
            return existing
        raise FileExistsError(f"prediction cache has different inputs: {output}")
    output.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = []; pending = {"issues": [], "low": [], "median": [], "high": [], "truth": [], "valid": [], "episodes": []}
    offset = 0; started = time.perf_counter()

    def flush():
        if not pending["issues"]: return
        issues = np.concatenate(pending["issues"])
        predictions = tuple(np.concatenate(pending[key]) for key in ("low", "median", "high"))
        truth = np.concatenate(pending["truth"]); valid = np.concatenate(pending["valid"])
        contexts = issue_contexts(dataset, issues, groups, context_mean, context_std)
        episodes=np.concatenate(pending["episodes"]) if dataset.episode_index is not None else None
        path = output/f"chunk-{len(chunks):04d}.npz"
        chunks.append(_write_chunk(path,issues,predictions,truth,valid,contexts,episodes))
        for value in pending.values(): value.clear()

    with torch.no_grad():
        for inputs, target, valid in loader:
            size = len(inputs); issues = dataset.issues[offset:offset+size]; offset += size
            low, median, high = model(inputs.to(device))
            real = tuple(value.cpu().numpy()*dataset.std+dataset.mean for value in (low, median, high))
            pending["issues"].append(issues); pending["low"].append(real[0]); pending["median"].append(real[1]); pending["high"].append(real[2])
            if dataset.episode_index is not None: pending["episodes"].append(dataset.episode_index[issues])
            target_rows=np.stack([dataset.values[int(issue)+1:int(issue)+dataset.horizon+1] for issue in issues])
            pending["truth"].append(target_rows); pending["valid"].append(valid.numpy())
            if sum(len(value) for value in pending["issues"]) >= chunk_origins: flush()
    flush()
    manifest = {"schema_version": SCHEMA_VERSION, "status": "completed", **input_fingerprint,
                "dataset": prepared_path.stem,
                "checkpoint_training_seed": int(saved["seed"]), "dataset_rows": len(dataset.values),
                "sensors": dataset.values.shape[1], "horizons": dataset.horizon,
                "origins": len(dataset), "units": "mph", "scenario": "C0_clean_inputs",
                "context_mean": context_mean.tolist(), "context_std": context_std.tolist(),
                "context_fit":"clean_training_histories_only","episode_aware":dataset.episode_index is not None,"chunks":chunks,
                "runtime_seconds": time.perf_counter()-started,
                "arrays": {"issue_bin": "int32 [origin]", "q05_mph/q50_mph/q95_mph": "float32 [origin,horizon,sensor]",
                           "truth_mph": "float32 [origin,horizon,sensor], NaN where invalid",
                           "original_valid": "bool [origin,horizon,sensor]", "context": "float32 [origin,sensor,2]"}}
    write_json(manifest_path, manifest)
    return manifest


def cache_faulted_predictions(prepared_path,checkpoint_path,fault_dir,output_dir,batch_size=32,
                              chunk_origins=288,device="cpu",partition="test"):
    """Cache predictions under one registered test input-release schedule."""
    prepared_path,checkpoint_path,fault_dir,output=map(Path,(prepared_path,checkpoint_path,fault_dir,output_dir))
    if partition not in {"tune","calibration","test","replication","confirmation","asym_validation"}: raise ValueError("unknown faulted cache partition")
    dataset=TrafficWindowDataset(prepared_path,partition,augment=False)
    with np.load(fault_dir/"release_schedule.npz") as schedule: input_arrival=schedule["input_arrival"].copy()
    fault_manifest_path=fault_dir/"fault_manifest.json"; fault_manifest=json.loads(fault_manifest_path.read_text(encoding="utf-8"))
    if fault_manifest.get("partition","test")!=partition: raise ValueError("fault schedule partition mismatch")
    test_start=int(fault_manifest.get("partition_start_row",fault_manifest["test_start_row"]))
    partition_length=int(fault_manifest.get("partition_length",fault_manifest["test_length"]))
    if input_arrival.shape!=(partition_length,dataset.values.shape[1]): raise ValueError("fault schedule shape mismatch")
    model,saved=load_checkpoint_model(checkpoint_path,dataset,device); groups=local_groups(dataset.adjacency)
    train=TrafficWindowDataset(prepared_path,"train",augment=False); context_mean,context_std,_=context_statistics(train)
    fingerprint={"prepared_sha256":sha256_file(prepared_path),"checkpoint_sha256":sha256_file(checkpoint_path),
                 "fault_manifest_sha256":sha256_file(fault_manifest_path),"release_schedule_sha256":sha256_file(fault_dir/"release_schedule.npz")}
    manifest_path=output/"manifest.json"
    if manifest_path.exists():
        existing=json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(existing.get(key)==value for key,value in fingerprint.items()):
            for chunk in existing["chunks"]:
                if sha256_file(output/chunk["file"])!=chunk["sha256"]: raise ValueError("faulted cache chunk hash mismatch")
            return existing
        raise FileExistsError(f"faulted prediction cache has different inputs: {output}")
    output.mkdir(parents=True,exist_ok=True); chunks=[]; started=time.perf_counter()
    for chunk_start in range(0,len(dataset.issues),chunk_origins):
        chunk_issues=dataset.issues[chunk_start:chunk_start+chunk_origins]
        lows=[]; medians=[]; highs=[]; truths=[]; target_valid=[]; contexts=[]
        for batch_start in range(0,len(chunk_issues),batch_size):
            issues=chunk_issues[batch_start:batch_start+batch_size]; feature_batch=[]; context_batch=[]
            for issue in issues:
                features,mask,age=faulted_history(dataset,input_arrival,test_start,issue); feature_batch.append(features)
                context_batch.append(np.asarray([((mask[:,group].mean()-context_mean[0])/context_std[0],
                                                  (np.log1p(age[:,group].mean())-context_mean[1])/context_std[1]) for group in groups],dtype=np.float32))
            with torch.no_grad(): low,median,high=model(torch.from_numpy(np.asarray(feature_batch)).to(device))
            real=tuple(value.cpu().numpy()*dataset.std+dataset.mean for value in (low,median,high))
            lows.append(real[0]); medians.append(real[1]); highs.append(real[2]); contexts.append(np.asarray(context_batch))
            for issue in issues:
                rows=slice(int(issue)+1,int(issue)+dataset.horizon+1); truths.append(dataset.values[rows]); target_valid.append(dataset.valid[rows])
        path=output/f"chunk-{len(chunks):04d}.npz"
        episodes=dataset.episode_index[chunk_issues] if dataset.episode_index is not None else None
        chunks.append(_write_chunk(path,chunk_issues,(np.concatenate(lows),np.concatenate(medians),np.concatenate(highs)),
                                   np.asarray(truths),np.asarray(target_valid),np.concatenate(contexts),episodes))
    manifest={"schema_version":SCHEMA_VERSION,"status":"completed",**fingerprint,"dataset":prepared_path.stem,
              "partition":partition,"scenario":fault_manifest["scenario_id"],"fault_seed":fault_manifest["seed"],
              "episode_aware":dataset.episode_index is not None,
              "checkpoint_training_seed":int(saved["seed"]),"sensors":dataset.values.shape[1],"horizons":dataset.horizon,
              "origins":len(dataset),"units":"mph","context_mean":context_mean.tolist(),"context_std":context_std.tolist(),
              "context_fit":"clean_training_histories_only","chunks":chunks,"runtime_seconds":time.perf_counter()-started,
              "arrays":{"issue_bin":"int32 [origin]","q05_mph/q50_mph/q95_mph":"float32 [origin,horizon,sensor]",
                        "truth_mph":"float32 [origin,horizon,sensor], NaN where invalid","original_valid":"bool [origin,horizon,sensor]",
                        "context":"float32 [origin,sensor,2]"}}
    write_json(manifest_path,manifest); return manifest


def compare_prediction_caches(reference_dir,candidate_dir,output_path):
    reference_dir,candidate_dir=Path(reference_dir),Path(candidate_dir)
    reference,reference_manifest=_read_cache_manifest(reference_dir)
    candidate,candidate_manifest=_read_cache_manifest(candidate_dir)
    if len(reference["chunks"])!=len(candidate["chunks"]): raise ValueError("cache chunk counts differ")
    rules={"issue_bin":("exact",0.),"original_valid":("exact",0.),
           "q05_mph":("exact",0.),"q50_mph":("exact",0.),"q95_mph":("exact",0.),
           "truth_mph":("tolerance",1e-5),"context":("tolerance",1e-6)}
    if reference.get("episode_aware") or candidate.get("episode_aware"):
        if not (reference.get("episode_aware") and candidate.get("episode_aware")):
            raise ValueError("cache episode-awareness differs")
        rules["episode_index"]=("exact",0.)
    comparisons={}
    for key,(rule,tolerance) in rules.items():
        exact=True; maximum=0.
        for left_meta,right_meta in zip(reference["chunks"],candidate["chunks"]):
            left_path=reference_dir/left_meta["file"]; right_path=candidate_dir/right_meta["file"]
            if sha256_file(left_path)!=left_meta["sha256"] or sha256_file(right_path)!=right_meta["sha256"]: raise ValueError("cache hash mismatch")
            with np.load(left_path) as left,np.load(right_path) as right:
                if np.issubdtype(left[key].dtype,np.floating):
                    exact &= np.array_equal(left[key],right[key],equal_nan=True)
                    difference=np.abs(left[key]-right[key]); maximum=max(maximum,float(np.nanmax(difference)))
                else: exact &= np.array_equal(left[key],right[key])
        comparisons[key]={"rule":rule,"exact":bool(exact),"max_abs_difference":maximum,"tolerance":tolerance,
                          "passes":bool(exact if rule=="exact" else maximum<=tolerance)}
    result={"schema_version":1,"status":"completed","reference_manifest_sha256":sha256_file(reference_manifest),
            "candidate_manifest_sha256":sha256_file(candidate_manifest),"comparisons":comparisons,
            "equivalent":all(value["passes"] for value in comparisons.values()),
            "note":"Truth tolerance covers legacy normalized float32 round-trip in the clean cache; model outputs require exact equality."}
    write_json(output_path,result); return result


def _read_cache_manifest(directory):
    path=Path(directory)/"manifest.json"
    return json.loads(path.read_text(encoding="utf-8")),path
