"""Registered observation-fault controls for the real-data stress study."""
from __future__ import annotations

import numpy as np

from ps1.online.faults import FaultSchedule, generate_fault_schedule


def matched_independent_control(source: FaultSchedule, seed: int) -> FaultSchedule:
    """Permute affected identities per bin, retaining each failed packet's draw.

    Event windows, affected-entry counts, and the multiset of arrival/loss
    outcomes at each time bin are unchanged. Independent per-bin permutations
    remove the persistent connected sensor group.
    """
    if source.scenario_id not in {"C3", "C4"}:
        raise ValueError("matched control requires a coupled C3 or C4 schedule")
    if not np.array_equal(source.input_arrival, source.feedback_arrival):
        raise ValueError("matched control requires identical input and feedback channels")
    length, nodes = source.input_arrival.shape
    base = np.broadcast_to(np.arange(length)[:, None], (length, nodes)).copy()
    result = base.copy()
    rng = np.random.default_rng(seed)
    events = []
    for event in source.events:
        start, end = int(event["start_bin"]), int(event["end_bin"])
        original = np.asarray(event["affected_sensors"], dtype=int)
        chosen = []
        for bin_index in range(start, end):
            # Keep the draw assigned to every failed entry, including -1 loss.
            draws = source.input_arrival[bin_index, original]
            new_sensors = rng.choice(nodes, size=len(original), replace=False)
            result[bin_index, new_sensors] = draws
            chosen.append(new_sensors.tolist())
        events.append({
            "event_index": int(event["event_index"]),
            "fault_type": "matched_independent_control",
            "source_scenario": source.scenario_id,
            "start_bin": start, "end_bin": end,
            "affected_sensor_count_per_bin": len(original),
            "affected_sensors_by_bin": chosen,
            "source_affected_sensors": original.tolist(),
        })
    return FaultSchedule(source.scenario_id, source.seed, result, result.copy(), tuple(events))


def one_factor_schedule(scenario: str, length: int, adjacency: np.ndarray, seed: int,
                        factor: str, value: float) -> FaultSchedule:
    """Change exactly one registered C3/C4 observation-fault parameter."""
    if scenario not in {"C3", "C4"}:
        raise ValueError("one-factor stress requires C3 or C4")
    kwargs = {"connected_fraction": .10, "duration_bins": 6,
              "backlog_release_bins": 3, "events_per_day": 2,
              "loss_within": 0. if scenario == "C3" else .50}
    if factor == "duration_bins":
        kwargs["duration_bins"] = int(value)
    elif factor == "connected_fraction":
        kwargs["connected_fraction"] = float(value)
    elif factor == "loss_within":
        kwargs["loss_within"] = float(value)
    elif factor != "additional_backlog_delay_bins":
        raise ValueError(f"unknown one-factor stress: {factor}")
    # The C4 generator records packet-level loss. Retain the requested C3/C4
    # identity even when the isolated loss parameter crosses its base value.
    generator_scenario = "C4" if kwargs["loss_within"] > 0 else "C3"
    schedule = generate_fault_schedule(generator_scenario, length, adjacency, seed, **kwargs)
    arrivals = schedule.input_arrival.copy()
    events = []
    extra = int(value) if factor == "additional_backlog_delay_bins" else 0
    for event in schedule.events:
        event = dict(event)
        if extra:
            bins = np.arange(event["start_bin"], event["end_bin"])
            sensors = np.asarray(event["affected_sensors"])
            affected = arrivals[np.ix_(bins, sensors)]
            arrivals[np.ix_(bins, sensors)] = np.where(affected >= 0, affected + extra, -1)
        event["additional_backlog_delay_bins"] = extra
        events.append(event)
    return FaultSchedule(scenario, seed, arrivals, arrivals.copy(), tuple(events))


def mnar_schedule(source: FaultSchedule, values: np.ndarray, thresholds: np.ndarray,
                  test_start_row: int, seed: int, additional_low_speed_loss: float) -> FaultSchedule:
    """Add evaluator-side loss during true low speed within fixed outage windows."""
    if source.scenario_id not in {"C3", "C4"}:
        raise ValueError("MNAR stress requires C3 or C4")
    result = source.input_arrival.copy()
    rng = np.random.default_rng(seed)
    events = []
    for event in source.events:
        event = dict(event)
        bins = np.arange(event["start_bin"], event["end_bin"])
        sensors = np.asarray(event["affected_sensors"], dtype=int)
        low = values[np.ix_(test_start_row + bins, sensors)] < thresholds[sensors]
        existing = result[np.ix_(bins, sensors)]
        additional = low & (existing >= 0) & (rng.random(low.shape) < additional_low_speed_loss)
        result[np.ix_(bins, sensors)] = np.where(additional, -1, existing)
        event["additional_mnar_lost_packets"] = int(additional.sum())
        event["low_speed_entries"] = int(low.sum())
        event["additional_low_speed_loss_probability"] = additional_low_speed_loss
        events.append(event)
    return FaultSchedule(source.scenario_id, source.seed, result, result.copy(), tuple(events))


def loss_variant_from_source(source: FaultSchedule, seed: int, loss_within: float) -> FaultSchedule:
    """Vary outage loss while retaining source groups and packet-delay draws."""
    if source.scenario_id not in {"C3", "C4"}:
        raise ValueError("loss stress requires C3 or C4")
    if not 0 <= loss_within <= 1:
        raise ValueError("loss fraction must be in [0, 1]")
    length, nodes = source.input_arrival.shape
    result = np.broadcast_to(np.arange(length)[:, None], (length, nodes)).copy()
    events = []
    for event in source.events:
        event = dict(event)
        bins = np.arange(event["start_bin"], event["end_bin"])
        sensors = np.asarray(event["affected_sensors"], dtype=int)
        delays = np.asarray(event["backlog_extra_delay"], dtype=int)
        if delays.shape != (len(bins), len(sensors)):
            raise ValueError("saved source packet-delay draws have wrong shape")
        arrivals = int(event["end_bin"]) + delays
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(event["event_index"])]))
        lost = rng.random(arrivals.shape) < loss_within
        result[np.ix_(bins, sensors)] = np.where(lost, -1, arrivals)
        event["fault_type"] = "correlated_loss_stress"
        event["loss_probability"] = loss_within
        event["lost_packets"] = int(lost.sum())
        events.append(event)
    return FaultSchedule(source.scenario_id, source.seed, result, result.copy(), tuple(events))
