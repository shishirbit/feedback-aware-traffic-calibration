from dataclasses import dataclass
from collections import defaultdict
import math


@dataclass(frozen=True)
class Forecast:
    forecast_id: str
    episode_id: str
    issue_bin: int
    target_bin: int
    horizon: int
    sensor_id: int
    point: float
    scale: float
    intervals: tuple
    context: tuple
    model_version: str = "unversioned"
    calibrator_version: str = "unversioned"

    def __post_init__(self):
        if self.horizon <= 0 or self.target_bin != self.issue_bin + self.horizon:
            raise ValueError("invalid forecast timestamps")
        if not math.isfinite(self.point) or not math.isfinite(self.scale) or self.scale < 1:
            raise ValueError("finite point and real-unit scale >= 1 required")
        if not isinstance(self.intervals, tuple) or not isinstance(self.context, tuple):
            raise ValueError("immutable tuple context and intervals required")
        for item in self.intervals:
            if not isinstance(item, tuple) or len(item) != 3:
                raise ValueError("interval must be (level, lower, upper)")
            level, lower, upper = item
            if not 0 < level < 1 or math.isnan(lower) or math.isnan(upper) or lower > upper:
                raise ValueError("invalid interval")


class ForecastLedger:
    def __init__(self):
        self.records = {}
        self._index = defaultdict(list)
        self._seen = set()
        self.feedback_matches = []

    def append(self, forecast):
        if forecast.forecast_id in self.records:
            raise ValueError("forecast ID already exists")
        self.records[forecast.forecast_id] = forecast
        self._index[(forecast.episode_id, forecast.sensor_id, forecast.target_bin)].append(forecast.forecast_id)

    def ingest_feedback(self, event, calibrator, now):
        if event.channel != "feedback" or event.arrival_bin > now:
            raise ValueError("not released feedback")
        if not event.original_valid:
            return
        key = (event.episode_id, event.sensor_id, event.observed_bin)
        for fid in self._index.get(key, ()):
            if fid in self._seen:
                continue
            forecast = self.records[fid]
            calibrator.observe(forecast, event, now)
            residual = abs(event.value - forecast.point) / forecast.scale
            misses = tuple((level, not lower <= event.value <= upper) for level, lower, upper in forecast.intervals)
            self.feedback_matches.append((fid, event.event_id, event.arrival_bin, forecast.target_bin, residual, misses))
            self._seen.add(fid)
