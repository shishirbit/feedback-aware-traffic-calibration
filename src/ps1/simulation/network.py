"""Deterministic synthetic motorway input generator for SUMO netconvert."""
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from ps1.utils.artifacts import sha256_file, write_json


def _write_xml(path, root):
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def build_network(output_dir, mainline_length=10000, stations=20, lanes=2):
    if stations != 20 or lanes != 2 or mainline_length <= 0:
        raise ValueError("registered network is 10 km/20 stations/two lanes")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    segment = mainline_length / stations
    nodes = ET.Element("nodes")
    for i in range(stations + 1):
        ET.SubElement(nodes, "node", id=f"n{i}", x=str(i * segment), y="0", type="priority")
    for label, i, y in (("on3", 6, -250), ("on7", 14, -250), ("off8", 16, 250)):
        ET.SubElement(nodes, "node", id=label, x=str(i * segment - (250 if label.startswith('on') else -250)), y=str(y), type="priority")
    _write_xml(output / "nodes.nod.xml", nodes)

    edges = ET.Element("edges")
    for i in range(stations):
        ET.SubElement(edges, "edge", id=f"m{i}", **{"from":f"n{i}", "to":f"n{i+1}", "numLanes":str(lanes), "speed":"27.78", "priority":"10"})
    ET.SubElement(edges, "edge", id="on3", **{"from":"on3", "to":"n6", "numLanes":"1", "speed":"16.67", "priority":"5"})
    ET.SubElement(edges, "edge", id="on7", **{"from":"on7", "to":"n14", "numLanes":"1", "speed":"16.67", "priority":"5"})
    ET.SubElement(edges, "edge", id="off8", **{"from":"n16", "to":"off8", "numLanes":"1", "speed":"16.67", "priority":"5"})
    _write_xml(output / "edges.edg.xml", edges)

    net = output / "motorway.net.xml"
    subprocess.run(["netconvert", "--node-files", str(output / "nodes.nod.xml"),
                    "--edge-files", str(output / "edges.edg.xml"), "--output-file", str(net),
                    "--no-turnarounds", "true"], check=True, capture_output=True, text=True)
    # netconvert writes a wall-clock timestamp in an XML comment. Re-serialize
    # the parsed document so identical inputs have byte-identical artifacts.
    net_root=ET.parse(net).getroot()
    ET.indent(net_root)
    ET.ElementTree(net_root).write(net,encoding="utf-8",xml_declaration=True)

    detectors = ET.Element("additional")
    for station in range(stations):
        for lane in range(lanes):
            # Negative positions are relative to the computed lane end. Junction
            # geometry can shorten a nominal 500 m edge at merges and exits.
            ET.SubElement(detectors, "inductionLoop", id=f"e1_{station}_{lane}", lane=f"m{station}_{lane}",
                          pos="-5", friendlyPos="true", period="300", file="detectors.xml")
    _write_xml(output / "detectors.add.xml", detectors)

    adjacency = np.zeros((stations, stations), dtype=np.float32)
    for i in range(stations - 1):
        adjacency[i, i + 1] = adjacency[i + 1, i] = 1
    np.savez_compressed(output / "adjacency.npz", adjacency=adjacency,
                        sensor_ids=np.array([f"station_{i}" for i in range(stations)]))
    manifest = {"kind":"synthetic_motorway", "mainline_length_m":mainline_length,
                "stations":stations, "through_lanes":lanes,
                "net_sha256":sha256_file(net), "detectors_sha256":sha256_file(output / "detectors.add.xml")}
    write_json(output / "network_manifest.json", manifest)
    return manifest
