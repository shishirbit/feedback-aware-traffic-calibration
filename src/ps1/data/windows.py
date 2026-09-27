import numpy as np


def split_bounds(length):
    return (0, int(.6 * length), int(.7 * length), int(.8 * length), length)


def origins(start, end, history=12, horizon=12):
    """Partition is [start,end); every target must stay inside it."""
    return range(max(history - 1, start - 1), end - horizon)


def training_statistics(values, original_valid, train_end):
    values = np.asarray(values, float)[:train_end]
    valid = np.asarray(original_valid, bool)[:train_end] & np.isfinite(values)
    masked = np.where(valid, values, np.nan)
    if not valid.any(axis=0).all():
        raise ValueError("each sensor needs valid training observations")
    return dict(mean=float(np.nanmean(masked)), std=max(float(np.nanstd(masked)), 1e-8),
                medians=np.nanmedian(masked, axis=0), congestion=np.nanpercentile(masked, 20, axis=0))
