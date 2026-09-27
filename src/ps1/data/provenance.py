from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from ps1.data.windows import split_bounds
from ps1.utils.artifacts import sha256_file


def audit_hdf(path, expected_sensors, source_url, graph_path=None, zero_is_missing=True,
              retrieval_url=None, license_access_terms=None):
    path=Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset unavailable: {path}")
    frame=pd.read_hdf(path)
    if frame.ndim!=2 or frame.shape[1]!=expected_sensors:
        raise ValueError(f"expected {expected_sensors} sensors, found {frame.shape}")
    index=pd.DatetimeIndex(frame.index)
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("timestamps must increase without duplicates")
    deltas_seconds=index.to_series().diff().dropna().dt.total_seconds()
    values=frame.to_numpy(dtype=float)
    finite=np.isfinite(values)
    sensor_ids=[str(value) for value in frame.columns]
    sensor_order_sha=hashlib.sha256("\n".join(sensor_ids).encode()).hexdigest()
    audit={
        "schema_version":2,
        "source_url":source_url,
        "retrieval_url":retrieval_url,
        "license_access_terms":license_access_terms,
        "retrieval_date":"2026-09-09",
        "path":str(path.resolve()),
        "sha256":sha256_file(path),
        "rows":len(frame),
        "sensors":frame.shape[1],
        "first_timestamp":index[0].isoformat(),
        "last_timestamp":index[-1].isoformat(),
        "timezone":str(index.tz) if index.tz is not None else None,
        "expected_frequency_minutes":5,
        "timestamp_gap_count":int((deltas_seconds!=300).sum()),
        "largest_timestamp_gap_minutes":float(deltas_seconds.max()/60),
        "finite_fraction":float(finite.mean()),
        "original_valid_fraction":float((finite & ((values!=0) if zero_is_missing else True)).mean()),
        "nan_count":int(np.isnan(values).sum()),
        "zero_count":int((finite&(values==0)).sum()),
        "negative_count":int((finite&(values<0)).sum()),
        "minimum_finite":float(values[finite].min()),
        "maximum_finite":float(values[finite].max()),
        "sensor_order_sha256":sensor_order_sha,
        "split_bounds":list(split_bounds(len(frame))),
        "target_units":"mph",
        "zero_is_missing":zero_is_missing,
        "missing_value_audit":"The pinned DCRNN evaluation uses null_val=0; benchmark zeros are treated as missing. SUMO retains physical zero speeds as valid.",
    }
    if graph_path:
        graph_path=Path(graph_path)
        audit["graph_path"]=str(graph_path.resolve())
        audit["graph_sha256"]=sha256_file(graph_path) if graph_path.exists() else None
    return audit
