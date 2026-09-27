"""Private evaluator-side generation of registered observation-release faults."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class FaultSchedule:
    scenario_id: str
    seed: int
    input_arrival: np.ndarray
    feedback_arrival: np.ndarray
    events: tuple

    def arrival(self,sensor_id,observed_bin,channel):
        array=self.input_arrival if channel=="input" else self.feedback_arrival
        value=int(array[observed_bin,sensor_id])
        return math.inf if value<0 else value

    def manifest(self):
        def encode(event):
            return {key:(value.tolist() if isinstance(value,np.ndarray) else value) for key,value in event.items()}
        return {"scenario_id":self.scenario_id,"seed":self.seed,"events":[encode(event) for event in self.events],
                "realized":{"input_unreleased_fraction":float((self.input_arrival<0).mean()),
                            "feedback_unreleased_fraction":float((self.feedback_arrival<0).mean()),
                            "input_delayed_fraction":float(((self.input_arrival>=0)&(self.input_arrival>np.arange(len(self.input_arrival))[:,None])).mean()),
                            "feedback_delayed_fraction":float(((self.feedback_arrival>=0)&(self.feedback_arrival>np.arange(len(self.feedback_arrival))[:,None])).mean())}}


def _connected_nodes(adjacency,size,rng):
    adjacency=np.asarray(adjacency)>0
    start=int(rng.integers(len(adjacency)))
    selected=[start]; seen={start}; frontier=[start]
    while frontier and len(selected)<size:
        node=frontier.pop(0)
        neighbors=[int(value) for value in np.flatnonzero(adjacency[node]) if int(value) not in seen]
        rng.shuffle(neighbors)
        for neighbor in neighbors:
            seen.add(neighbor); selected.append(neighbor); frontier.append(neighbor)
            if len(selected)==size: break
    if len(selected)<size:
        # A disconnected graph is explicit: fill deterministically from remaining
        # nodes and report connected=false in the event manifest.
        selected.extend(node for node in range(len(adjacency)) if node not in seen and len(selected)<size)
    return np.asarray(sorted(selected),dtype=int),len(seen)>=size


def generate_fault_schedule(scenario_id,length,adjacency,seed,connected_fraction=.10,duration_bins=6,
                            backlog_release_bins=3,events_per_day=2,independent_loss=.10,loss_within=.50,
                            event_starts=None):
    if scenario_id not in {"C0","C1","C2","C3","C4","C5"}:
        raise ValueError("scenario must be C0-C5")
    nodes=len(adjacency); base=np.broadcast_to(np.arange(length)[:,None],(length,nodes)).copy()
    inputs=base.copy(); feedback=base.copy(); events=[]; rng=np.random.default_rng(seed)
    if scenario_id=="C5":
        lost=rng.random((length,nodes))<independent_loss
        inputs[lost]=feedback[lost]=-1
        events.append({"fault_type":"independent_loss","probability":independent_loss,"lost_entries":int(lost.sum())})
        return FaultSchedule(scenario_id,seed,inputs,feedback,tuple(events))
    if scenario_id=="C0":
        return FaultSchedule(scenario_id,seed,inputs,feedback,())
    affected_count=max(1,round(nodes*connected_fraction))
    # Two fixed, widely separated daily windows avoid post-selection while seeds
    # determine graph location and packet-level backlog/loss draws.
    if event_starts is None:
        offsets=(72,216) if events_per_day==2 else tuple(np.linspace(48,240,events_per_day,dtype=int))
        scheduled=(day*288+int(offset) for day in range(length//288) for offset in offsets)
    else:
        scheduled=(int(value) for value in event_starts)
    for start in scheduled:
            if start<0 or start+duration_bins>length: raise ValueError("fault event falls outside stream")
            end=start+duration_bins
            sensors,connected=_connected_nodes(adjacency,affected_count,rng)
            packet_bins=np.arange(start,end)
            delay=rng.integers(0,backlog_release_bins,size=(len(packet_bins),len(sensors)))
            arrival=end+delay
            event={"event_index":len(events),"fault_type":"correlated_buffered" if scenario_id!="C4" else "correlated_lossy",
                   "start_bin":start,"end_bin":end,"affected_sensors":sensors,"realized_fraction":len(sensors)/nodes,
                   "graph_connected":connected,"backlog_extra_delay":delay}
            target_inputs=scenario_id in {"C2","C3","C4"}
            target_feedback=scenario_id in {"C1","C3","C4"}
            if scenario_id=="C4":
                lost=rng.random(arrival.shape)<loss_within
                arrival=np.where(lost,-1,arrival); event["loss_probability"]=loss_within; event["lost_packets"]=int(lost.sum())
            ix=np.ix_(packet_bins,sensors)
            if target_inputs: inputs[ix]=arrival
            if target_feedback: feedback[ix]=arrival
            events.append(event)
    return FaultSchedule(scenario_id,seed,inputs,feedback,tuple(events))
