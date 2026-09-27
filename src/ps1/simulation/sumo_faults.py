"""Episode-reset reporting faults for held-out SUMO streams."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from ps1.online.faults import generate_fault_schedule
from ps1.utils.artifacts import sha256_file,write_json


def generate_sumo_faults(prepared_path,prepared_manifest_path,scenario,seed,output_dir,config,partition_name="test"):
    if scenario not in {"C0","C1","C2","C3","C4"}: raise ValueError("SUMO core supports C0-C4")
    prepared_path,prepared_manifest_path,output=Path(prepared_path),Path(prepared_manifest_path),Path(output_dir)
    with np.load(prepared_path) as data:
        episode_index=data["episode_index"]; partition=data["partition_code"]; adjacency=data["adjacency"]
    partition_codes={"tune":1,"test":3,"replication":4,"confirmation":4,"asym_validation":4}
    if partition_name not in partition_codes: raise ValueError("unknown SUMO fault partition")
    code=partition_codes[partition_name]; test_episodes=np.unique(episode_index[partition==code]); test_start=int(np.flatnonzero(partition==code)[0])
    inputs=[]; feedback=[]; events=[]; offset=0; affected=[]
    faults=config["faults"]
    for test_ordinal,episode in enumerate(test_episodes):
        rows=np.flatnonzero(episode_index==episode); length=len(rows)
        # Two events per full equivalent day: one event in two of each three 8-hour episodes.
        starts=(48,) if test_ordinal%3!=2 and scenario!="C0" else ()
        episode_seed=int(np.random.SeedSequence([seed,int(episode)]).generate_state(1)[0])
        schedule=generate_fault_schedule(scenario,length,adjacency,episode_seed,
            faults["connected_fraction"],faults["duration_bins"],faults["backlog_release_bins"],
            faults["events_per_day"],event_starts=starts)
        for array,target in ((schedule.input_arrival,inputs),(schedule.feedback_arrival,feedback)):
            target.append(np.where(array>=0,array+offset,-1))
        if schedule.events: affected.append(int(episode))
        for event in schedule.events:
            encoded={key:(value.tolist() if isinstance(value,np.ndarray) else value) for key,value in event.items()}
            encoded.update({"episode_index":int(episode),"test_episode_ordinal":test_ordinal,
                            "local_start_bin":encoded["start_bin"],"local_end_bin":encoded["end_bin"],
                            "start_bin":encoded["start_bin"]+offset,"end_bin":encoded["end_bin"]+offset})
            events.append(encoded)
        offset+=length
    input_arrival=np.concatenate(inputs); feedback_arrival=np.concatenate(feedback)
    output.mkdir(parents=True,exist_ok=True); schedule_path=output/"release_schedule.npz"
    if schedule_path.exists():
        with np.load(schedule_path) as prior:
            if not (np.array_equal(prior["input_arrival"],input_arrival) and np.array_equal(prior["feedback_arrival"],feedback_arrival)):
                raise FileExistsError(f"differing SUMO fault schedule: {schedule_path}")
    else: np.savez_compressed(schedule_path,input_arrival=input_arrival,feedback_arrival=feedback_arrival)
    bins=np.arange(len(input_arrival))[:,None]
    manifest={"schema_version":1,"scenario_id":scenario,"seed":seed,"dataset":"sumo","episodic":True,"partition":partition_name,
              "partition_start_row":test_start,"partition_length":len(input_arrival),"episode_indices":test_episodes.tolist(),
              "test_start_row":test_start,"test_length":len(input_arrival),"test_episode_indices":test_episodes.tolist(),
              "affected_episode_indices":affected,"event_count":len(events),"events":events,
              "event_rate_rule":"one event in two of each three ordered 8-hour episodes; equivalent to two events per 24 hours",
              "prepared_sha256":sha256_file(prepared_path),"prepared_manifest_sha256":sha256_file(prepared_manifest_path),
              "release_schedule_sha256":sha256_file(schedule_path),"realized":{
              "input_unreleased_fraction":float((input_arrival<0).mean()),"feedback_unreleased_fraction":float((feedback_arrival<0).mean()),
              "input_delayed_fraction":float(((input_arrival>=0)&(input_arrival>bins)).mean()),
              "feedback_delayed_fraction":float(((feedback_arrival>=0)&(feedback_arrival>bins)).mean())}}
    write_json(output/"fault_manifest.json",manifest); return manifest


def derive_immediate_feedback_oracle(source_dir,output_dir):
    """Keep C4 point-input releases fixed while releasing every outcome immediately."""
    source,output=Path(source_dir),Path(output_dir)
    source_manifest_path=source/"fault_manifest.json"
    manifest=json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if manifest["scenario_id"]!="C4": raise ValueError("feedback oracle source must be C4")
    with np.load(source/"release_schedule.npz") as schedule:
        input_arrival=schedule["input_arrival"].copy()
    feedback_arrival=np.broadcast_to(np.arange(len(input_arrival),dtype=input_arrival.dtype)[:,None],input_arrival.shape).copy()
    output.mkdir(parents=True,exist_ok=True); schedule_path=output/"release_schedule.npz"
    if schedule_path.exists():
        with np.load(schedule_path) as prior:
            if not (np.array_equal(prior["input_arrival"],input_arrival) and np.array_equal(prior["feedback_arrival"],feedback_arrival)):
                raise FileExistsError(f"differing SUMO feedback oracle: {schedule_path}")
    else: np.savez_compressed(schedule_path,input_arrival=input_arrival,feedback_arrival=feedback_arrival)
    derived={key:value for key,value in manifest.items() if key not in {"scenario_id","release_schedule_sha256","realized"}}
    bins=np.arange(len(input_arrival))[:,None]
    derived.update({"scenario_id":"C4I","source_scenario_id":"C4","oracle_role":"same C4 inputs with immediate complete feedback",
        "source_fault_manifest_sha256":sha256_file(source_manifest_path),"release_schedule_sha256":sha256_file(schedule_path),"realized":{
        "input_unreleased_fraction":float((input_arrival<0).mean()),"feedback_unreleased_fraction":0.0,
        "input_delayed_fraction":float(((input_arrival>=0)&(input_arrival>bins)).mean()),"feedback_delayed_fraction":0.0}})
    write_json(output/"fault_manifest.json",derived); return derived
