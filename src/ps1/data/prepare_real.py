from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from ps1.data.windows import split_bounds
from ps1.utils.artifacts import sha256_file,write_json


def load_dcrnn_graph(path,expected_sensors):
    """Load a graph only from the pinned, hashed DCRNN provenance artifact."""
    with Path(path).open("rb") as handle:
        sensor_ids,index,adjacency=pickle.load(handle,encoding="latin1")
    adjacency=np.asarray(adjacency,dtype=np.float32)
    sensor_ids=[str(value) for value in sensor_ids]
    if adjacency.shape!=(expected_sensors,expected_sensors) or len(sensor_ids)!=expected_sensors:
        raise ValueError("DCRNN graph dimensions do not match expected sensors")
    if set(index)!=set(sensor_ids) or any(index[str(sensor)]!=i for i,sensor in enumerate(sensor_ids)):
        raise ValueError("DCRNN graph sensor index is inconsistent")
    return sensor_ids,adjacency


def prepare_hdf(hdf_path,graph_path,output_path,expected_sensors):
    hdf_path,graph_path,output_path=map(Path,(hdf_path,graph_path,output_path))
    frame=pd.read_hdf(hdf_path)
    if frame.shape[1]!=expected_sensors:
        raise ValueError("unexpected dataset dimensions")
    raw_index=pd.DatetimeIndex(frame.index)
    bounds=split_bounds(len(frame))
    boundary_times=[raw_index[i].isoformat() if i<len(raw_index) else (raw_index[-1]+np.timedelta64(5,"m")).isoformat() for i in bounds]
    full_index=pd.date_range(raw_index[0],raw_index[-1],freq="5min")
    inserted=~full_index.isin(raw_index)
    frame=frame.reindex(full_index)
    values=frame.to_numpy(dtype=np.float32)
    original_valid=np.isfinite(values)&(values!=0)
    sensor_ids,adjacency=load_dcrnn_graph(graph_path,expected_sensors)
    columns=[str(value) for value in frame.columns]
    if columns!=sensor_ids:
        raise ValueError("HDF sensor order differs from graph sensor order")
    train_end=pd.Timestamp(boundary_times[1])
    train_rows=full_index<train_end
    observed=np.where(original_valid[train_rows],values[train_rows],np.nan)
    if np.isnan(observed).all(axis=0).any():
        raise ValueError("at least one sensor lacks valid training values")
    mean=float(np.nanmean(observed)); std=max(float(np.nanstd(observed)),1e-8)
    medians=np.nanmedian(observed,axis=0).astype(np.float32)
    congestion=np.nanpercentile(observed,20,axis=0).astype(np.float32)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output_path,values_mph=values,original_valid=original_valid,
                        timestamp_ns=full_index.astype("int64"),sensor_ids=np.asarray(sensor_ids),adjacency=adjacency,
                        train_mean=np.float32(mean),train_std=np.float32(std),
                        train_sensor_median=medians,train_congestion_p20=congestion)
    manifest={"schema_version":1,"source_sha256":sha256_file(hdf_path),"graph_sha256":sha256_file(graph_path),
              "artifact_sha256":sha256_file(output_path),"rows":len(full_index),"raw_rows":len(raw_index),
              "inserted_missing_rows":int(inserted.sum()),"sensors":expected_sensors,"boundary_timestamps":boundary_times,
              "units":"mph","zero_is_missing":True,"arrays":{
                  "values_mph":"float32 [time,sensor]","original_valid":"bool [time,sensor]",
                  "timestamp_ns":"int64 naive nanoseconds","sensor_ids":"unicode [sensor]",
                  "adjacency":"float32 [sensor,sensor]"}}
    write_json(output_path.with_suffix(".manifest.json"),manifest)
    return manifest
