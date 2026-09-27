"""Producer-side gateway; consumers receive only release_due output."""
import heapq
import math
from .events import ReleaseEvent


class ReportingGateway:
    def __init__(self, release_policy):
        self._policy = release_policy
        self._queue = []
        self._counter = 0
        self._ingested = set()
        self._now = -math.inf

    def ingest_truth(self, observation):
        key = (observation.episode_id, observation.sensor_id, observation.observed_bin)
        if key in self._ingested:
            raise ValueError("duplicate producer observation")
        events = []
        for channel in ("input", "feedback"):
            arrival = self._policy(observation, channel)
            if arrival == math.inf:
                continue
            if not isinstance(arrival, int):
                raise ValueError("release time must be an integer bin or infinity")
            if arrival <= self._now:
                raise ValueError("cannot enqueue into an already released boundary")
            events.append(ReleaseEvent(
                f"{key[0]}:{key[1]}:{key[2]}:{channel}", channel,
                observation.episode_id, observation.observed_bin, arrival,
                observation.sensor_id, observation.value, observation.original_valid))
        self._ingested.add(key)
        for event in events:
            heapq.heappush(self._queue, (event.arrival_bin, self._counter, event))
            self._counter += 1

    def release_due(self, now):
        if now < self._now:
            raise ValueError("gateway clock cannot run backwards")
        self._now = now
        out = []
        while self._queue and self._queue[0][0] <= now:
            out.append(heapq.heappop(self._queue)[2])
        return sorted(out, key=lambda e: (e.arrival_bin, e.observed_bin, e.sensor_id, e.channel, e.event_id))
