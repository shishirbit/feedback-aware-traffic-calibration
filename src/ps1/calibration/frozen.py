import math
import numpy as np


def conformal_quantile(scores, alpha):
    scores = np.asarray(scores, dtype=float)
    if not 0 < alpha < 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("finite nonempty scores and alpha in (0,1) required")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return float("inf") if rank > len(scores) else float(np.sort(scores)[rank - 1])


def weighted_quantile(scores, weights, probability):
    scores, weights = np.asarray(scores, float), np.asarray(weights, float)
    if scores.shape != weights.shape or scores.ndim != 1:
        raise ValueError("scores and weights must be equal-length vectors")
    if not 0 < probability <= 1 or not np.isfinite(scores).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("invalid weighted distribution")
    keep = weights > 0
    scores, weights = scores[keep], weights[keep]
    if not len(scores):
        raise ValueError("empty weighted distribution")
    order = np.argsort(scores, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = min(np.searchsorted(cumulative, probability * cumulative[-1], side="left"), len(scores) - 1)
    return float(scores[order[index]])
