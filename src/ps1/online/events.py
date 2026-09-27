from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Observation:
    episode_id: str
    sensor_id: int
    observed_bin: int
    value: float
    original_valid: bool = True


@dataclass(frozen=True)
class ReleaseEvent:
    event_id: str
    channel: str
    episode_id: str
    observed_bin: int
    arrival_bin: int
    sensor_id: int
    value: float
    original_valid: bool = True

    def __post_init__(self):
        if self.channel not in ("input", "feedback"):
            raise ValueError("channel must be input or feedback")
        if self.arrival_bin < self.observed_bin:
            raise ValueError("arrival cannot precede observation")
        if self.original_valid and not math.isfinite(self.value):
            raise ValueError("valid observations must be finite")
