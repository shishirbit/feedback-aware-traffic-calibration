import numpy as np


def moving_block_mean_ci(paired_differences,block_bins=288,resamples=2000,seed=0):
    values=np.asarray(paired_differences,float)
    if values.ndim<1 or len(values)<block_bins or resamples<=0:
        raise ValueError("insufficient time bins or invalid resample count")
    if not np.isfinite(values).any():
        raise ValueError("paired differences contain no finite targets")
    flattened=values.reshape(len(values),-1); finite=np.isfinite(flattened)
    per_time_sum=np.nansum(flattened,axis=1); per_time_count=finite.sum(axis=1)
    rng=np.random.default_rng(seed); starts=np.arange(len(values)-block_bins+1)
    estimates=np.empty(resamples); block_count=int(np.ceil(len(values)/block_bins))
    for draw in range(resamples):
        chosen=rng.choice(starts,size=block_count,replace=True)
        indices=(chosen[:,None]+np.arange(block_bins)[None,:]).reshape(-1)[:len(values)]
        estimates[draw]=per_time_sum[indices].sum()/per_time_count[indices].sum()
    return {"estimate":float(per_time_sum.sum()/per_time_count.sum()),"lower":float(np.quantile(estimates,.025)),
            "upper":float(np.quantile(estimates,.975)),"resamples":resamples,"block_bins":block_bins,"seed":seed}


def episode_mean_ci(paired_episode_differences,resamples=2000,seed=0):
    values=np.asarray(paired_episode_differences,float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite episode-level paired differences required")
    rng=np.random.default_rng(seed)
    estimates=values[rng.integers(0,len(values),size=(resamples,len(values)))].mean(axis=1)
    return {"estimate":float(values.mean()),"lower":float(np.quantile(estimates,.025)),
            "upper":float(np.quantile(estimates,.975)),"resamples":resamples,"episodes":len(values),"seed":seed}
