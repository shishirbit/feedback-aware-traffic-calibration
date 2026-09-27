import numpy as np


class InputHistory:
    def __init__(self, episode_id, training_medians, start_bin=0):
        self.episode_id = episode_id
        self.medians = np.asarray(training_medians, dtype=float)
        if not np.isfinite(self.medians).all():
            raise ValueError("training medians must be finite")
        self.start_bin = start_bin
        self._values = [{} for _ in self.medians]

    def ingest(self, event, now):
        if event.channel != "input" or event.episode_id != self.episode_id:
            raise ValueError("wrong history channel or episode")
        if event.arrival_bin > now:
            raise ValueError("unreleased observation")
        if event.original_valid:
            self._values[event.sensor_id][event.observed_bin] = (event.value, event.arrival_bin)

    def causal_window(self, now, length=12):
        values = np.zeros((length, len(self.medians)))
        mask = np.zeros_like(values, dtype=bool)
        age = np.zeros_like(values)
        for row, slot in enumerate(range(now - length + 1, now + 1)):
            for sensor, received in enumerate(self._values):
                usable = [u for u, (_, a) in received.items() if u <= slot and a <= now]
                latest = max(usable) if usable else None
                values[row, sensor] = self.medians[sensor] if latest is None else received[latest][0]
                mask[row, sensor] = latest == slot
                age[row, sensor] = max(0, slot - self.start_bin + 1) if latest is None else slot - latest
        return values, mask, age
