import numpy as np


def interval_metrics(truth, point, lower, upper, original_valid, alpha):
    y, p, lo, hi = [np.asarray(x, float) for x in (truth, point, lower, upper)]
    valid = np.asarray(original_valid, bool)
    if not (y.shape == p.shape == lo.shape == hi.shape == valid.shape):
        raise ValueError("metric shapes differ")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    y, p, lo, hi = (x[valid] for x in (y, p, lo, hi))
    if not len(y):
        return {"count": 0, "status": "no valid targets"}
    if not np.isfinite(y).all() or not np.isfinite(p).all() or np.isnan(lo).any() or np.isnan(hi).any() or (hi < lo).any():
        raise ValueError("invalid scored values")
    score = hi - lo
    below, above = y < lo, y > hi
    score[below] += 2 / alpha * (lo[below] - y[below])
    score[above] += 2 / alpha * (y[above] - hi[above])
    coverage = float(np.mean((lo <= y) & (y <= hi)))
    return dict(count=len(y), mae=float(np.mean(abs(y-p))), rmse=float(np.sqrt(np.mean((y-p)**2))),
                picp=coverage, mpiw=float(np.mean(hi-lo)), interval_score=float(np.mean(score)),
                coverage_error=coverage-(1-alpha), undercoverage=max(0.,1-alpha-coverage))
