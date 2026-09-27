from pathlib import Path
import shutil
import subprocess
import uuid
import xml.etree.ElementTree as ET
import numpy as np
from ps1.utils.artifacts import sha256_file, write_json

MPH_PER_MPS = 2.2369362920544


def write_config(network_dir, episode_dir, episode_seconds, seed):
    network_dir,episode_dir=Path(network_dir),Path(episode_dir)
    # SUMO resolves an additional file's output relative to that additional
    # file, so keep an episode-local copy to isolate episode artifacts.
    detector_config=episode_dir / "detectors.add.xml"
    shutil.copyfile(network_dir / "detectors.add.xml",detector_config)
    root=ET.Element("configuration")
    inputs=ET.SubElement(root,"input")
    ET.SubElement(inputs,"net-file",value=str((network_dir / "motorway.net.xml").resolve()))
    ET.SubElement(inputs,"route-files",value=str((episode_dir / "routes.rou.xml").resolve()))
    ET.SubElement(inputs,"additional-files",value=str(detector_config.resolve()))
    time=ET.SubElement(root,"time")
    ET.SubElement(time,"begin",value="0"); ET.SubElement(time,"end",value=str(episode_seconds)); ET.SubElement(time,"step-length",value="1")
    processing=ET.SubElement(root,"processing")
    ET.SubElement(processing,"time-to-teleport",value="-1")
    random=ET.SubElement(root,"random_number")
    ET.SubElement(random,"seed",value=str(seed))
    ET.indent(root)
    path=Path(episode_dir,"episode.sumocfg")
    ET.ElementTree(root).write(path,encoding="utf-8",xml_declaration=True)
    return path


def run_offline(config_path, episode_dir, physical_events=()):
    episode_dir=Path(episode_dir)
    tripinfo=episode_dir / "tripinfo.xml"
    event_result={}
    if physical_events:
        import traci
        label=f"ps1-offline-{uuid.uuid4()}"
        traci.start(["sumo","-c",str(Path(config_path).resolve()),"--tripinfo-output",str(tripinfo.resolve()),"--no-warnings","true","--no-step-log","true"],label=label)
        connection=traci.getConnection(label)
        applied=[]
        try:
            end=connection.simulation.getEndTime()
            while connection.simulation.getTime()<end:
                now=int(connection.simulation.getTime())
                for index,event in enumerate(physical_events):
                    action=None
                    if now==event["start_s"]: action="restrict"
                    elif now==event["end_s"]: action="restore"
                    if action:
                        speed=event["speed_mps"] if action=="restrict" else event["restore_mps"]
                        lanes=connection.edge.getLaneNumber(event["edge"])
                        for lane in range(lanes):
                            connection.lane.setMaxSpeed(f"{event['edge']}_{lane}",speed)
                        applied.append({"event_index":index,"time_s":now,"action":action,
                                        "edge":event["edge"],"lanes":lanes,"speed_mps":speed})
                connection.simulationStep()
        finally:
            connection.close()
        if len(applied)!=2*len(physical_events):
            raise RuntimeError(f"applied {len(applied)} physical-event actions; expected {2*len(physical_events)}")
        event_log=episode_dir/"physical_event_log.json"
        write_json(event_log,{"requested":list(physical_events),"applied":applied})
        event_result={"physical_event_log_sha256":sha256_file(event_log),"physical_event_actions":len(applied)}
    else:
        result=subprocess.run(["sumo","-c",str(config_path),"--tripinfo-output",str(tripinfo),"--no-warnings","true"],cwd=episode_dir,capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError(f"SUMO failed with exit {result.returncode}:\n{result.stderr.strip()}")
    detector_path=episode_dir / "detectors.xml"
    if not detector_path.exists():
        raise RuntimeError("SUMO produced no detector output")
    return {"detectors_sha256":sha256_file(detector_path),"tripinfo_sha256":sha256_file(tripinfo),**event_result}


def parse_e1(path, stations=20, lanes=2):
    bins={}
    for _,element in ET.iterparse(path,events=("end",)):
        if element.tag!="interval":
            continue
        detector=element.attrib["id"].split("_")
        station,lane=int(detector[-2]),int(detector[-1])
        key=(float(element.attrib["begin"]),float(element.attrib["end"]),station)
        count=int(element.attrib.get("nVehContrib",0))
        speed=float(element.attrib.get("speed","-1"))
        bins.setdefault(key,[]).append((lane,count,speed))
        element.clear()
    keys=sorted(bins)
    values=np.full((len({(a,b) for a,b,_ in keys}),stations),np.nan)
    counts=np.zeros_like(values,dtype=np.int32)
    intervals=sorted({(a,b) for a,b,_ in keys})
    row_for={key:i for i,key in enumerate(intervals)}
    for (begin,end,station),rows in bins.items():
        total=sum(count for _,count,speed in rows if count>0 and speed>=0)
        counts[row_for[(begin,end)],station]=total
        if total:
            values[row_for[(begin,end)],station]=sum(count*speed for _,count,speed in rows if count>0 and speed>=0)/total*MPH_PER_MPS
    return np.array(intervals),values,counts
