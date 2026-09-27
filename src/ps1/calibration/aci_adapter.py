"""Arrival-aware spatial batching adapter for multi-step ACI.

This is a registered project variant of Hallberg Szabadvary (2024), Eq. (8).
The original result updates one error per horizon and time step. Here, feedback
may arrive late and several sensors may arrive together, so one update uses the
mean miss indicator in each (horizon, coverage-level) arrival batch. The
original finite-sample guarantee therefore is not claimed for this adapter.
"""
from collections import defaultdict
from copy import deepcopy

import numpy as np

from ps1.calibration.far_cal import Residual
from ps1.calibration.frozen import conformal_quantile


class ArrivalBatchACI:
    def __init__(self, groups, learning_rates=None, buffer_bins=2016):
        self.groups = tuple(tuple(group) for group in groups)
        self.levels = (.90, .95)
        rates = learning_rates or {.90: .005, .95: .005}
        if set(rates) != set(self.levels) or any(rate <= 0 for rate in rates.values()):
            raise ValueError("positive learning rate required for each registered level")
        self.learning_rates = {float(level): float(rate) for level, rate in rates.items()}
        self.buffer_bins = int(buffer_bins)
        if self.buffer_bins <= 0:
            raise ValueError("positive buffer required")
        self.live = []
        self.seen = set()
        self.control = {}
        self.pending = defaultdict(list)
        self.update_count = defaultdict(int)

    @property
    def version(self):
        return "arrival-batch-aci-v1"

    def initialize(self, records):
        records = list(records)
        if not records:
            raise ValueError("initial calibration archive is empty")
        if len({record.forecast_id for record in records}) != len(records):
            raise ValueError("duplicate archive residual")
        if any(record.score < 0 or record.arrival_bin < record.target_bin for record in records):
            raise ValueError("invalid calibration record")
        self.live = records.copy()
        self.seen = {record.forecast_id for record in records}
        for horizon in {record.horizon for record in records}:
            for level in self.levels:
                self.control[(horizon, level)] = 1. - level

    def observe(self, forecast, event, now):
        if (event.channel != "feedback" or event.arrival_bin > now
                or event.observed_bin != forecast.target_bin
                or event.sensor_id != forecast.sensor_id
                or event.episode_id != forecast.episode_id):
            raise ValueError("feedback does not causally match forecast")
        if not event.original_valid or forecast.forecast_id in self.seen:
            return
        self.live.append(Residual(
            forecast.forecast_id, forecast.sensor_id, forecast.horizon,
            forecast.target_bin, event.arrival_bin,
            abs(event.value - forecast.point) / forecast.scale, forecast.context,
        ))
        issued = {float(level): (lower, upper) for level, lower, upper in forecast.intervals}
        for level in self.levels:
            if level not in issued:
                raise ValueError(f"issued forecast lacks {level:.2f} interval")
            lower, upper = issued[level]
            self.pending[(forecast.horizon, level)].append(float(not lower <= event.value <= upper))
        self.seen.add(forecast.forecast_id)

    def finish_arrival_batch(self, now):
        del now  # Arrival eligibility is checked by observe.
        for key, misses in self.pending.items():
            horizon, level = key
            target_alpha = 1. - level
            alpha = self.control.setdefault(key, target_alpha)
            self.control[key] = alpha + self.learning_rates[level] * (target_alpha - float(np.mean(misses)))
            self.update_count[key] += 1
        self.pending.clear()

    def expire_by_target_time(self, now):
        self.live = [record for record in self.live if record.target_bin >= now - self.buffer_bins]

    def _records(self, now, sensor, horizon):
        eligible = [record for record in self.live
                    if record.horizon == horizon and record.arrival_bin <= now and record.target_bin <= now]
        local = [record for record in eligible if record.sensor_id == sensor]
        return (local if len(local) >= 50 else eligible), len(local) < 50

    def predict_one(self, now, sensor, horizon, point, scale, context, missing_fraction):
        del context
        if not 0 <= missing_fraction <= 1 or scale < 1 or not np.isfinite([point, scale]).all():
            raise ValueError("invalid forecast or observable missingness")
        records, global_fallback = self._records(now, sensor, horizon)
        if not records:
            raise ValueError(f"no received residuals for horizon {horizon}")
        scores = np.asarray([record.score for record in records], dtype=float)
        intervals = []
        active = {}
        for level in self.levels:
            alpha = self.control.setdefault((horizon, level), 1. - level)
            active[level] = alpha
            if alpha <= 0:
                width = float("inf")
            elif alpha >= 1:
                width = 0.
            else:
                width = scale * conformal_quantile(scores, alpha)
            lower = max(0., point - width)
            intervals.append((level, lower, max(lower, point + width)))
        freshest = max(record.target_bin for record in records)
        return tuple(intervals), {
            "method": "arrival_batch_aci",
            "control_alpha": active,
            "records": len(records),
            "global_fallback": global_fallback,
            "freshest_target_age": max(0, now - freshest),
            "missing_fraction_ignored": missing_fraction,
            "guarantee": "project_variant_no_inherited_finite_sample_claim",
        }

    def state_dict(self):
        state = deepcopy(self.__dict__)
        state["pending"] = dict(state["pending"])
        state["update_count"] = dict(state["update_count"])
        return state

    def load_state_dict(self, state):
        state = deepcopy(state)
        state["pending"] = defaultdict(list, state["pending"])
        state["update_count"] = defaultdict(int, state["update_count"])
        self.__dict__ = state
