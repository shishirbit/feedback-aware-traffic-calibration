"""TraCI interval collection and paired offline-output equivalence check."""
from pathlib import Path
import json
import uuid
import numpy as np
import traci
from ps1.simulation.runner import MPH_PER_MPS,parse_e1


def run_live(config_path,episode_dir,stations=20,lanes=2,physical_events=()):
    label=f"ps1-{uuid.uuid4()}"
    traci.start(["sumo","-c",str(Path(config_path).resolve()),"--no-warnings","true"],label=label)
    connection=traci.getConnection(label)
    collected=[]
    try:
        end=connection.simulation.getEndTime()
        while connection.simulation.getTime()<end:
            now=int(connection.simulation.getTime())
            for event in physical_events:
                if now==event["start_s"]:
                    for lane in range(lanes): connection.lane.setMaxSpeed(f"{event['edge']}_{lane}",event["speed_mps"])
                if now==event["end_s"]:
                    for lane in range(lanes): connection.lane.setMaxSpeed(f"{event['edge']}_{lane}",event["restore_mps"])
            connection.simulationStep()
            now=int(connection.simulation.getTime())
            if now>0 and now%300==0:
                speed=np.full(stations,np.nan); count=np.zeros(stations,dtype=np.int32)
                for station in range(stations):
                    lane_rows=[]
                    for lane in range(lanes):
                        detector=f"e1_{station}_{lane}"
                        lane_rows.append((connection.inductionloop.getLastIntervalVehicleNumber(detector),
                                          connection.inductionloop.getLastIntervalMeanSpeed(detector)))
                    total=sum(n for n,s in lane_rows if n>0 and s>=0); count[station]=total
                    if total: speed[station]=sum(n*s for n,s in lane_rows if n>0 and s>=0)/total*MPH_PER_MPS
                collected.append((now,speed,count))
    finally:
        connection.close()
    live_speed=np.stack([row[1] for row in collected]); live_count=np.stack([row[2] for row in collected])
    intervals,offline_speed,offline_count=parse_e1(Path(episode_dir)/"detectors.xml",stations,lanes)
    if live_speed.shape!=offline_speed.shape:
        raise AssertionError(f"live/offline shapes differ: {live_speed.shape} vs {offline_speed.shape}")
    same_missing=np.array_equal(np.isnan(live_speed),np.isnan(offline_speed))
    max_error=float(np.nanmax(np.abs(live_speed-offline_speed))) if np.isfinite(live_speed).any() else 0.
    counts_equal=bool(np.array_equal(live_count,offline_count))
    # XML stores speed to 0.01 m/s; 0.012 mph bounds the corresponding
    # 0.005 m/s rounding difference after lane-weighted aggregation.
    tolerance=.012
    return {"bins":len(collected),"same_missing_mask":same_missing,"counts_equal":counts_equal,
            "max_speed_error_mph":max_error,"tolerance_mph":tolerance,
            "equivalent":bool(same_missing and counts_equal and max_error<=tolerance),
            "tolerance_basis":"SUMO XML speed precision is 0.01 m/s; TraCI retains greater precision"}
