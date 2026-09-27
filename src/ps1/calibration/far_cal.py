"""Exact reference FAR-Cal; intended for validation before scaling optimization."""
from dataclasses import dataclass
from copy import deepcopy
import numpy as np
from .frozen import weighted_quantile


@dataclass(frozen=True)
class Residual:
    forecast_id: str
    sensor_id: int
    horizon: int
    target_bin: int
    arrival_bin: int
    score: float
    context: tuple


class FARCal:
    def __init__(self, groups, tau=288., bandwidth=1., support=50., stale_decay=288.,
                 beta_gap=.25, beta_missing=.25, inflation_cap=3., buffer_bins=2016):
        if min(tau, bandwidth, support, stale_decay, buffer_bins) <= 0:
            raise ValueError("positive decay, bandwidth, support and buffer required")
        self.groups = tuple(tuple(g) for g in groups)
        self.tau, self.bandwidth, self.support = tau, bandwidth, support
        self.stale_decay = stale_decay
        self.beta_gap, self.beta_missing = beta_gap, beta_missing
        self.inflation_cap, self.buffer_bins = inflation_cap, buffer_bins
        self.frozen, self.live, self.seen = [], [], set()

    @property
    def version(self):
        return "far-cal-exact-v1"

    def initialize(self, calibration_records):
        records = list(calibration_records)
        if not records:
            raise ValueError("initial calibration archive is empty")
        if any(r.arrival_bin < r.target_bin or r.score < 0 or not np.isfinite(r.score) for r in records):
            raise ValueError("invalid calibration record")
        if len({r.forecast_id for r in records}) != len(records):
            raise ValueError("duplicate initial residual")
        self.frozen = records.copy()
        self.live = records.copy()
        self.seen = {r.forecast_id for r in records}

    def observe(self, forecast, event, now):
        if event.channel != "feedback" or event.arrival_bin > now or event.observed_bin != forecast.target_bin or event.sensor_id != forecast.sensor_id or event.episode_id != forecast.episode_id:
            raise ValueError("feedback does not causally match forecast")
        if not event.original_valid or forecast.forecast_id in self.seen:
            return
        self.live.append(Residual(forecast.forecast_id, forecast.sensor_id, forecast.horizon,
                                  forecast.target_bin, event.arrival_bin,
                                  abs(event.value - forecast.point) / forecast.scale, forecast.context))
        self.seen.add(forecast.forecast_id)

    def finish_arrival_batch(self, now):
        pass  # No order-dependent adaptive parameter steps in FAR-Cal.

    def expire_by_target_time(self, now):
        self.live = [r for r in self.live if r.target_bin >= now - self.buffer_bins]

    def _pool(self, records, now, context):
        if not records:
            return np.array([]), np.array([]), 0.
        targets = np.array([r.target_bin for r in records])
        distances = np.sum((np.array([r.context for r in records]) - context) ** 2, axis=1)
        log_weights = -(now - targets) / self.tau - distances / (2 * self.bandwidth ** 2)
        weights = np.exp(log_weights - log_weights.max())
        _, bins = np.unique(targets, return_inverse=True)
        blocks = np.bincount(bins, weights=weights)
        ess = float(blocks.sum() ** 2 / np.square(blocks).sum())
        return np.array([r.score for r in records]), weights / weights.sum(), ess

    def predict_one(self, now, sensor, horizon, point, scale, context, missing_fraction):
        if not 0 <= missing_fraction <= 1 or scale < 1 or not np.isfinite([point, scale]).all():
            raise ValueError("invalid forecast or observable missingness")
        archive = [r for r in self.frozen if r.horizon == horizon]
        if any(r.arrival_bin > now or r.target_bin > now for r in archive):
            raise ValueError("initial archive contains future feedback")
        node_archive = [r for r in archive if r.sensor_id == sensor]
        frozen = node_archive if len(node_archive) >= 50 else archive
        if not frozen:
            raise ValueError(f"no frozen residuals for horizon {horizon}; initialize calibration first")
        eligible = [r for r in self.live if r.horizon == horizon and r.arrival_bin <= now and now - self.buffer_bins <= r.target_bin <= now]
        local = [r for r in eligible if r.sensor_id in self.groups[sensor]]
        ls, lw, le = self._pool(local, now, np.asarray(context))
        gs, gw, ge = self._pool(eligible, now, np.asarray(context))
        age_archive = [r for r in archive if r.sensor_id in self.groups[sensor]]
        freshest = local or age_archive or archive
        gap = max(0, now - max(r.target_bin for r in freshest))
        eta = min(1., ge / self.support) * np.exp(-gap / self.stale_decay) if len(gs) else 0.
        mix = le / (le + self.support) if len(ls) else 0.
        scores = np.concatenate([ls, gs, [r.score for r in frozen]])
        weights = np.concatenate([eta * mix * lw, eta * (1 - mix) * gw, np.full(len(frozen), (1 - eta) / len(frozen))])
        inflation = min(self.inflation_cap, 1 + self.beta_gap * min(gap / self.stale_decay, 2) + self.beta_missing * missing_fraction)
        intervals = []
        clipped = []
        for level in (.90, .95):
            width = scale * weighted_quantile(scores, weights, level) * inflation
            lower = max(0., point - width)
            intervals.append((level, lower, max(lower, point + width)))
            clipped.append(point - width < 0)
        return tuple(intervals), dict(local_ess=le, global_ess=ge, gap=gap, eta=float(eta),
                                     inflation=float(inflation), archive_global=len(node_archive) < 50,
                                     age_global=not bool(local or age_archive), lower_clipped=clipped)

    def state_dict(self):
        return deepcopy(self.__dict__)

    def load_state_dict(self, state):
        self.__dict__ = deepcopy(state)
