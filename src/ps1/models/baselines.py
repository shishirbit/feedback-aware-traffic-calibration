import numpy as np


class LastAvailable:
    """Sanity control; scale is a training-only sensor statistic."""
    version = "last-available-v1"

    def __init__(self, scales, horizons=12):
        self.scales = np.maximum(1., np.asarray(scales, float))
        self.horizons = horizons

    def predict(self, history, mask, age, issue_bin=None):
        del mask, age, issue_bin
        point = np.broadcast_to(history[-1], (self.horizons, len(self.scales))).copy()
        scale = np.broadcast_to(self.scales, point.shape).copy()
        return point, scale, None
