"""Prepare paper-scale SUMO episodes without crossing episode boundaries."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from ps1.utils.artifacts import sha256_file, write_json


PARTITION_CODE={"train":0,"tune":1,"calibration":2,"test":3}


def prepare_sumo_paper(paper_root, output_path, warmup_bins=12):
    paper_root,output_path=Path(paper_root),Path(output_path)
    generation_path=paper_root/"generation_manifest.json"
    generation=json.loads(generation_path.read_text(encoding="utf-8"))
    if generation.get("status")!="completed" or len(generation.get("results",[]))!=120:
        raise ValueError("complete 120-episode SUMO generation manifest required")
    rows=sorted(generation["results"],key=lambda x:(list(PARTITION_CODE).index(x["split"]),x["index"]))
    values=[]; valid=[]; timestamps=[]; episode_index=[]; partition_code=[]; truth_hashes=[]; offsets=[0]
    base=np.datetime64("2000-01-03T00:00","ns")
    for ordinal,row in enumerate(rows):
        summary=Path(row["artifact_path"])
        truth=summary.parent/"detector_truth.npz"
        with np.load(truth) as data:
            speed=data["speed_mph"].astype(np.float32)
            observed=data["original_valid"].astype(bool)&np.isfinite(speed)
            intervals=data["intervals"]
        if speed.shape!=(96,20) or intervals.shape!=(96,2):
            raise ValueError(f"unexpected episode arrays: {truth}")
        values.append(speed); valid.append(observed)
        timestamps.append((base+ordinal*np.timedelta64(1,"D")+np.arange(96)*np.timedelta64(5,"m")).astype("int64"))
        episode_index.append(np.full(96,ordinal,dtype=np.int16))
        partition_code.append(np.full(96,PARTITION_CODE[row["split"]],dtype=np.uint8))
        truth_hashes.append({"episode_id":row["episode_id"],"sha256":sha256_file(truth)})
        offsets.append(offsets[-1]+96)
    values=np.concatenate(values); valid=np.concatenate(valid); partitions=np.concatenate(partition_code)
    train=partitions==PARTITION_CODE["train"]
    observed=np.where(valid[train],values[train],np.nan)
    if np.isnan(observed).all(axis=0).any():
        raise ValueError("at least one SUMO sensor lacks valid training values")
    with np.load(paper_root/"network"/"adjacency.npz") as graph:
        adjacency=graph["adjacency"].astype(np.float32); sensor_ids=graph["sensor_ids"]
    output_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output_path,values_mph=values,original_valid=valid,
                        timestamp_ns=np.concatenate(timestamps),sensor_ids=sensor_ids,adjacency=adjacency,
                        train_mean=np.float32(np.nanmean(observed)),train_std=np.float32(max(np.nanstd(observed),1e-8)),
                        train_sensor_median=np.nanmedian(observed,axis=0).astype(np.float32),
                        train_congestion_p20=np.nanpercentile(observed,20,axis=0).astype(np.float32),
                        episode_index=np.concatenate(episode_index),partition_code=partitions,
                        warmup_bins=np.int16(warmup_bins),
                        episode_ids=np.asarray([row["episode_id"] for row in rows]),
                        episode_family=np.asarray([row["family"] for row in rows]),
                        episode_offsets=np.asarray(offsets,dtype=np.int32))
    manifest={"schema_version":1,"source_generation_sha256":sha256_file(generation_path),
              "artifact_sha256":sha256_file(output_path),"rows":len(values),"episodes":len(rows),"sensors":20,
              "bins_per_episode":96,"warmup_bins":int(warmup_bins),"split_episode_counts":dict(Counter(x["split"] for x in rows)),
              "family_episode_counts":dict(Counter(x["family"] for x in rows)),"truth_inputs":truth_hashes,
              "synthetic_clock":"one episode per successive day from 2000-01-03; 16-hour inter-episode gap",
              "units":"mph","arrays":{"episode_index":"int16 [time], resets causal state",
              "partition_code":"uint8 [time], train/tune/calibration/test = 0/1/2/3",
              "episode_offsets":"int32 [episodes+1]"}}
    write_json(output_path.with_suffix(".manifest.json"),manifest)
    return manifest


def prepare_sumo_replication(replication_root,reference_path,output_path):
    replication_root,reference_path,output_path=map(Path,(replication_root,reference_path,output_path))
    generation_path=replication_root/"generation_manifest.json"; generation=json.loads(generation_path.read_text(encoding="utf-8"))
    if generation.get("status")!="completed" or not generation.get("results"):
        raise ValueError("complete independent replication manifest required")
    rows=sorted(generation["results"],key=lambda row:row["index"]); values=[]; valid=[]; truth_hashes=[]
    for row in rows:
        truth=Path(row["artifact_path"]).parent/"detector_truth.npz"
        with np.load(truth) as data:
            speed=data["speed_mph"].astype(np.float32); observed=data["original_valid"].astype(bool)&np.isfinite(speed)
        if speed.shape!=(96,20): raise ValueError(f"unexpected replication arrays: {truth}")
        values.append(speed); valid.append(observed); truth_hashes.append({"episode_id":row["episode_id"],"sha256":sha256_file(truth)})
    with np.load(reference_path) as reference:
        original={key:reference[key].copy() for key in reference.files}
    with np.load(replication_root/"network"/"adjacency.npz") as graph:
        if not np.array_equal(original["sensor_ids"],graph["sensor_ids"]) or not np.allclose(original["adjacency"],graph["adjacency"]):
            raise ValueError("replication network differs from training network")
    count=len(rows); first_episode=len(original["episode_ids"]); day_ns=86_400_000_000_000; starts=original["timestamp_ns"][-96]+day_ns
    new_timestamps=np.concatenate([starts+i*day_ns+np.arange(96)*300_000_000_000 for i in range(count)])
    result={**original,"values_mph":np.concatenate((original["values_mph"],np.concatenate(values))),
        "original_valid":np.concatenate((original["original_valid"],np.concatenate(valid))),
        "timestamp_ns":np.concatenate((original["timestamp_ns"],new_timestamps)),
        "episode_index":np.concatenate((original["episode_index"],np.repeat(np.arange(first_episode,first_episode+count,dtype=np.int16),96))),
        "partition_code":np.concatenate((original["partition_code"],np.full(count*96,4,dtype=np.uint8))),
        "episode_ids":np.concatenate((original["episode_ids"],np.asarray([row["episode_id"] for row in rows]))),
        "episode_family":np.concatenate((original["episode_family"],np.asarray([row["family"] for row in rows]))),
        "episode_offsets":np.arange(first_episode+count+1,dtype=np.int32)*96}
    output_path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(output_path,**result)
    manifest={"schema_version":1,"status":"completed_independent_replication","reference_prepared_sha256":sha256_file(reference_path),
        "source_generation_sha256":sha256_file(generation_path),"artifact_sha256":sha256_file(output_path),"rows":len(result["values_mph"]),
        "episodes":first_episode+count,"replication_episode_indices":list(range(first_episode,first_episode+count)),"replication_truth_inputs":truth_hashes,
        "partition_codes":{"train":0,"tune":1,"calibration":2,"test":3,"replication":4}}
    write_json(output_path.with_suffix(".manifest.json"),manifest); return manifest
