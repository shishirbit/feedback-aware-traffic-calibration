"""Deterministic, resumable generation of the registered SUMO paper episodes."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np

from ps1.simulation.demand import generate_routes
from ps1.simulation.network import build_network
from ps1.simulation.runner import parse_e1, run_offline, write_config
from ps1.utils.artifacts import write_json


SPLIT_COUNTS = {"train": 60, "tune": 15, "calibration": 15, "test": 30}
FAMILY_COUNTS = {
    "train": {"P0": 15, "P1": 15, "P2": 15, "P3": 15},
    "tune": {"P0": 4, "P1": 4, "P2": 4, "P3": 3},
    "calibration": {"P0": 4, "P1": 4, "P2": 3, "P3": 4},
    "test": {"P0": 7, "P1": 7, "P2": 8, "P3": 8},
}
SEED_BASE = {"train": 10000, "tune": 20000, "calibration": 30000, "test": 40000}


def _interleave(counts, offset=0):
    remaining=dict(counts); families=["P0","P1","P2","P3"]
    order=families[offset:]+families[:offset]; result=[]
    while sum(remaining.values()):
        for family in order:
            if remaining[family]:
                result.append(family); remaining[family]-=1
    return result


def paper_episode_plan():
    plan=[]
    for offset,(split,count) in enumerate(SPLIT_COUNTS.items()):
        families=_interleave(FAMILY_COUNTS[split],offset)
        if len(families)!=count:
            raise AssertionError(f"invalid registered count for {split}")
        for index,family in enumerate(families,1):
            plan.append({"episode_id":f"{split}-{index:03d}","split":split,
                         "index":index,"seed":SEED_BASE[split]+index,"family":family})
    return plan


def _generate_one(task):
    item,spec,network,output_root,config_sha256=task
    episode=Path(output_root)/item.get("directory",item["split"])/item["episode_id"]
    summary_path=episode/"run_summary.json"
    if summary_path.exists():
        prior=json.loads(summary_path.read_text(encoding="utf-8"))
        expected=(item["seed"],item["family"],config_sha256)
        actual=(prior.get("seed"),prior.get("family"),prior.get("resolved_config_sha256"))
        if actual!=expected:
            raise RuntimeError(f"existing episode metadata mismatch: {summary_path}")
        return {**item,"status":"existing","artifact_path":str(summary_path)}
    generate_routes(network,episode,spec["episode_seconds"],item["seed"],item["family"])
    sumocfg=write_config(network,episode,spec["episode_seconds"],item["seed"])
    manifest=json.loads((episode/"episode_manifest.json").read_text(encoding="utf-8"))
    hashes=run_offline(sumocfg,episode,manifest["physical_events"])
    intervals,speeds,counts=parse_e1(episode/"detectors.xml",spec["stations"],spec["through_lanes"])
    np.savez_compressed(episode/"detector_truth.npz",intervals=intervals,speed_mph=speeds,
                        nVehContrib=counts,original_valid=np.isfinite(speeds))
    summary={**hashes,"bins":len(intervals),"valid_fraction":float(np.isfinite(speeds).mean()),
             "units":"mph","seed":item["seed"],"family":item["family"],"split":item["split"],
             "episode_id":item["episode_id"],"resolved_config_sha256":config_sha256}
    write_json(summary_path,summary)
    return {**item,"status":"generated","artifact_path":str(summary_path)}


def generate_paper_dataset(config, config_sha256, output_root, workers=3):
    output_root=Path(output_root); network=output_root/"network"
    build_network(network,config["sumo"]["mainline_length_m"],config["sumo"]["stations"],config["sumo"]["through_lanes"])
    plan=paper_episode_plan()
    plan_manifest={"schema_version":1,"resolved_config_sha256":config_sha256,
                   "episode_seconds":config["sumo"]["episode_seconds"],"episodes":plan,
                   "split_counts":dict(Counter(x["split"] for x in plan)),
                   "family_counts":dict(Counter(x["family"] for x in plan))}
    write_json(output_root/"episode_plan.json",plan_manifest)
    tasks=[(item,config["sumo"],str(network),str(output_root),config_sha256) for item in plan]
    completed=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(_generate_one,task) for task in tasks]
        for future in as_completed(futures):
            completed.append(future.result())
            print(f"completed {len(completed)}/{len(plan)}",flush=True)
    completed.sort(key=lambda x:(list(SPLIT_COUNTS).index(x["split"]),x["index"]))
    manifest={**plan_manifest,"status":"completed","workers":workers,"results":completed}
    write_json(output_root/"generation_manifest.json",manifest)
    return manifest


def replication_episode_plan(registry=None):
    registry=registry or {"family_counts":{"P0":7,"P1":7,"P2":8,"P3":8},"episode_seed_base":50000,"episode_id_prefix":"replication"}
    counts=registry["family_counts"]; prefix=registry.get("episode_id_prefix","replication"); seed_base=registry["episode_seed_base"]
    return [{"episode_id":f"{prefix}-{index:03d}","split":"replication","directory":"episodes",
             "index":index,"seed":seed_base+index,"family":family}
            for index,family in enumerate(_interleave(counts,3),1)]


def generate_replication_dataset(config,config_sha256,output_root,workers=3,registry=None):
    output_root=Path(output_root); network=output_root/"network"
    build_network(network,config["sumo"]["mainline_length_m"],config["sumo"]["stations"],config["sumo"]["through_lanes"])
    plan=replication_episode_plan(registry)
    plan_manifest={"schema_version":1,"status":"registered_independent_replication","resolved_config_sha256":config_sha256,
        "episode_seconds":config["sumo"]["episode_seconds"],"episodes":plan,"split_counts":{"replication":len(plan)},
        "family_counts":dict(Counter(x["family"] for x in plan))}
    write_json(output_root/"episode_plan.json",plan_manifest)
    tasks=[(item,config["sumo"],str(network),str(output_root),config_sha256) for item in plan]
    completed=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(_generate_one,task) for task in tasks]
        for future in as_completed(futures):
            completed.append(future.result()); print(f"completed {len(completed)}/{len(plan)}",flush=True)
    completed.sort(key=lambda x:x["index"])
    manifest={**plan_manifest,"status":"completed","workers":workers,"results":completed}
    write_json(output_root/"generation_manifest.json",manifest); return manifest
