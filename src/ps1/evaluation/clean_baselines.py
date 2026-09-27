"""Streaming clean-input point, raw-quantile, and frozen-conformal evaluation."""
from pathlib import Path
import json
import math

import numpy as np

from ps1.calibration.frozen import conformal_quantile
from ps1.data.windows import origins,split_bounds
from ps1.training import _causal_state
from ps1.utils.artifacts import sha256_file, write_json


def _read_manifest(directory):
    path=Path(directory)/"manifest.json"
    return json.loads(path.read_text(encoding="utf-8")),path


def frozen_quantiles(calibration_dir, levels=(.90,.95)):
    manifest,_=_read_manifest(calibration_dir)
    shape=(manifest["origins"],manifest["horizons"],manifest["sensors"])
    scores=np.empty(shape,dtype=np.float32); valid=np.empty(shape,dtype=bool); offset=0
    for chunk in manifest["chunks"]:
        path=Path(calibration_dir)/chunk["file"]
        if sha256_file(path)!=chunk["sha256"]: raise ValueError("calibration chunk hash mismatch")
        with np.load(path) as data:
            size=len(data["issue_bin"]); scale=np.maximum((data["q95_mph"]-data["q05_mph"])/2,1.)
            scores[offset:offset+size]=np.abs(data["truth_mph"]-data["q50_mph"])/scale
            valid[offset:offset+size]=data["original_valid"]
            offset+=size
    quantiles=np.empty((len(levels),shape[1],shape[2]),dtype=np.float32)
    counts=np.empty((shape[1],shape[2]),dtype=np.int32)
    for horizon in range(shape[1]):
        for sensor in range(shape[2]):
            values=scores[:,horizon,sensor][valid[:,horizon,sensor]]; counts[horizon,sensor]=len(values)
            for index,level in enumerate(levels): quantiles[index,horizon,sensor]=conformal_quantile(values,1-level)
    return quantiles,counts


class _Sums:
    def __init__(self):
        self.count=0; self.abs_error=0.; self.square_error=0.; self.ape=0.; self.mape_count=0
        self.covered=0; self.width=0.; self.interval_score=0.

    def point(self,truth,point,valid):
        error=(truth-point)[valid]; self.count+=len(error); self.abs_error+=np.abs(error).sum(); self.square_error+=np.square(error).sum()
        mape=valid&(truth>=5); self.ape+=(np.abs(truth-point)[mape]/truth[mape]).sum(); self.mape_count+=int(mape.sum())

    def interval(self,truth,lower,upper,valid,alpha):
        y=truth[valid]; lo=lower[valid]; hi=upper[valid]
        self.count+=len(y); self.covered+=((lo<=y)&(y<=hi)).sum(); self.width+=(hi-lo).sum()
        self.interval_score+=((hi-lo)+(2/alpha)*(lo-y)*(y<lo)+(2/alpha)*(y-hi)*(y>hi)).sum()

    def point_result(self):
        return {"count":int(self.count),"mae_mph":float(self.abs_error/self.count),"rmse_mph":float(math.sqrt(self.square_error/self.count)),
                "mape_percent":float(100*self.ape/self.mape_count),"mape_excluded_fraction":float(1-self.mape_count/self.count)}

    def interval_result(self):
        return {"count":int(self.count),"picp":float(self.covered/self.count),"mpiw_mph":float(self.width/self.count),
                "mean_interval_score":float(self.interval_score/self.count)}


def evaluate_clean_baselines(calibration_dir,test_dir,output_path,report_horizons=(3,6,12)):
    calibration,cal_manifest_path=_read_manifest(calibration_dir); test,test_manifest_path=_read_manifest(test_dir)
    if calibration["checkpoint_sha256"]!=test["checkpoint_sha256"]: raise ValueError("cache checkpoints differ")
    quantiles,archive_counts=frozen_quantiles(calibration_dir)
    accumulators={}
    for h in report_horizons:
        accumulators[("point",h,None)]=_Sums(); accumulators[("raw",h,.90)]=_Sums()
        for level in (.90,.95): accumulators[("frozen",h,level)]=_Sums()
    for chunk in test["chunks"]:
        path=Path(test_dir)/chunk["file"]
        if sha256_file(path)!=chunk["sha256"]: raise ValueError("test chunk hash mismatch")
        with np.load(path) as data:
            truth=data["truth_mph"]; point=data["q50_mph"]; valid=data["original_valid"]
            scale=np.maximum((data["q95_mph"]-data["q05_mph"])/2,1.)
            for h in report_horizons:
                index=h-1; mask=valid[:,index]
                accumulators[("point",h,None)].point(truth[:,index],point[:,index],mask)
                accumulators[("raw",h,.90)].interval(truth[:,index],np.maximum(0,data["q05_mph"][:,index]),data["q95_mph"][:,index],mask,.10)
                for level_index,level in enumerate((.90,.95)):
                    width=scale[:,index]*quantiles[level_index,index][None,:]
                    lower=np.maximum(0,point[:,index]-width); upper=np.maximum(lower,point[:,index]+width)
                    accumulators[("frozen",h,level)].interval(truth[:,index],lower,upper,mask,1-level)
    point={str(h):accumulators[("point",h,None)].point_result() for h in report_horizons}
    intervals={"raw_q05_q95":{str(h):accumulators[("raw",h,.90)].interval_result() for h in report_horizons},
               "frozen":{str(level):{str(h):accumulators[("frozen",h,level)].interval_result() for h in report_horizons} for level in (.90,.95)}}
    intervals["primary_equal_horizon_mean_frozen_90_interval_score"]=float(np.mean([intervals["frozen"]["0.9"][str(h)]["mean_interval_score"] for h in report_horizons]))
    inferred_dataset=Path(test_dir).parent.name.rsplit("-seed",1)[0]
    scenario=test.get("scenario","C0_clean_inputs"); clean=scenario in {"C0","C0_clean_inputs"}
    limitations=["No online adaptation or feedback-fault comparison","Single training seed"]
    if clean: limitations.insert(0,"Clean C0 input/feedback only")
    else: limitations.insert(0,f"Static raw/frozen evaluation under {scenario} inputs; feedback schedule does not affect these methods")
    result={"schema_version":1,"status":"completed_clean_C0_only" if clean else "completed_static_fault","dataset":test.get("dataset",inferred_dataset),
            "scenario":scenario,"fault_seed":test.get("fault_seed"),
            "training_seed":test["checkpoint_training_seed"],"checkpoint_sha256":test["checkpoint_sha256"],
            "calibration_manifest_sha256":sha256_file(cal_manifest_path),"test_manifest_sha256":sha256_file(test_manifest_path),
            "report_horizons":list(report_horizons),"point":point,"intervals":intervals,
            "archive_count_min":int(archive_counts.min()),"archive_count_max":int(archive_counts.max()),
            "limitations":limitations}
    write_json(output_path,result); return result


def evaluate_simple_controls(prepared_path,output_path,report_horizons=(3,6,12)):
    """Evaluate registered point controls with training-only statistics."""
    prepared_path=Path(prepared_path)
    with np.load(prepared_path) as data:
        values=data["values_mph"].astype(np.float32); valid=data["original_valid"].astype(bool)&np.isfinite(values)
        timestamps=data["timestamp_ns"]; medians=data["train_sensor_median"].astype(np.float32)
    bounds=split_bounds(len(values)); train_end=bounds[1]
    stamps=timestamps.astype("datetime64[ns]"); day=stamps.astype("datetime64[D]")
    minute=(stamps-day).astype("timedelta64[m]").astype(np.int64); weekday=(day.astype(np.int64)+3)%7
    week_slot=weekday*288+minute//5
    historical=np.broadcast_to(medians,(2016,len(medians))).copy()
    for slot in np.unique(week_slot[:train_end]):
        indices=np.flatnonzero(week_slot[:train_end]==slot)
        rows_valid=valid[indices]; candidates=np.where(rows_valid,values[indices],np.nan)
        present=rows_valid.any(axis=0); historical[int(slot),present]=np.nanmedian(candidates[:,present],axis=0)
    filled,_=_causal_state(values,valid,medians)
    test_issues=np.asarray(list(origins(bounds[3],bounds[4])),dtype=np.int64)
    methods={name:{h:_Sums() for h in report_horizons} for name in ("last_available","historical_time_of_week")}
    for h in report_horizons:
        targets=test_issues+h; truth=values[targets]; mask=valid[targets]
        last=filled[test_issues]; seasonal=historical[week_slot[targets]]
        methods["last_available"][h].point(truth,last,mask)
        methods["historical_time_of_week"][h].point(truth,seasonal,mask)
    result={"schema_version":1,"status":"completed_clean_C0_only","dataset":prepared_path.stem,
            "prepared_sha256":sha256_file(prepared_path),"report_horizons":list(report_horizons),
            "methods":{name:{str(h):accumulator.point_result() for h,accumulator in values_by_h.items()} for name,values_by_h in methods.items()},
            "rules":{"last_available":"most recent causally observed speed at issue; training sensor median before any observation",
                     "historical_time_of_week":"training median by actual weekday and 5-minute slot; training sensor-median fallback"},
            "limitations":["Clean C0 inputs only","Point controls do not produce uncertainty intervals"]}
    write_json(output_path,result); return result
