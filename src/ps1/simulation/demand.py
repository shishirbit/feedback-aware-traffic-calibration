from pathlib import Path
import random
import xml.etree.ElementTree as ET
from ps1.utils.artifacts import sha256_file, write_json


def generate_routes(network_dir, output_dir, episode_seconds, seed, family="P0"):
    if family not in {"P0", "P1", "P2", "P3"}:
        raise ValueError("unknown physical family")
    rng = random.Random(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    routes = ET.Element("routes")
    ET.SubElement(routes,"vType",id="passenger",vClass="passenger",length="5",accel="2.6",decel="4.5",sigma="0.5",maxSpeed="27.78",color="0,0,1")
    ET.SubElement(routes,"vType",id="truck",vClass="truck",length="12",accel="1.3",decel="3.5",sigma="0.5",maxSpeed="22.22",color="1,0,0")
    ET.SubElement(routes,"vTypeDistribution",id="mix",vTypes="passenger truck",probs="0.9 0.1")
    ET.SubElement(routes,"route",id="main",edges=" ".join(f"m{i}" for i in range(20)))
    ET.SubElement(routes,"route",id="from_on3",edges="on3 " + " ".join(f"m{i}" for i in range(6,20)))
    ET.SubElement(routes,"route",id="from_on7",edges="on7 " + " ".join(f"m{i}" for i in range(14,20)))
    ET.SubElement(routes,"route",id="main_exit",edges=" ".join([*(f"m{i}" for i in range(16)),"off8"]))
    # Separate explicit flows preserve attempted departures in the generated manifest.
    split = int(episode_seconds * .45)
    peak = family in {"P1","P3"}
    factor = rng.uniform(1.4,1.8) if peak else 1.0
    ET.SubElement(routes,"flow",id="main_before",type="mix",route="main",begin="0",end=str(split),vehsPerHour="1620",departLane="best",departSpeed="max")
    ET.SubElement(routes,"flow",id="exit_before",type="mix",route="main_exit",begin="0",end=str(split),vehsPerHour="180",departLane="best",departSpeed="max")
    ET.SubElement(routes,"flow",id="main_after",type="mix",route="main",begin=str(split),end=str(episode_seconds),vehsPerHour=str(1620*factor),departLane="best",departSpeed="max")
    ET.SubElement(routes,"flow",id="exit_after",type="mix",route="main_exit",begin=str(split),end=str(episode_seconds),vehsPerHour=str(180*factor),departLane="best",departSpeed="max")
    for ramp in ("on3","on7"):
        ET.SubElement(routes,"flow",id=ramp,type="mix",route=f"from_{ramp}",begin="0",end=str(episode_seconds),vehsPerHour=str(150*factor),departLane="best",departSpeed="max")
    ET.indent(routes)
    route_path=output / "routes.rou.xml"
    ET.ElementTree(routes).write(route_path,encoding="utf-8",xml_declaration=True)
    schedule=[]
    if family in {"P2","P3"}:
        start=rng.randint(7200,max(7200,episode_seconds-7200-2400))
        schedule=[{"edge":"m15","start_s":start,"end_s":start+rng.randint(1200,2400),"speed_mps":8.33,"restore_mps":27.78}]
    write_json(output / "episode_manifest.json",{"seed":seed,"family":family,"episode_seconds":episode_seconds,
               "demand_factor":factor,"physical_events":schedule,"routes_sha256":sha256_file(route_path)})
    return route_path
