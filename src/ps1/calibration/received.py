"""Received-feedback-only interval comparators sharing FAR-Cal's interface."""
from copy import deepcopy
import numpy as np
from ps1.calibration.far_cal import Residual
from ps1.calibration.frozen import conformal_quantile,weighted_quantile


class ReceivedResidualCalibrator:
    def __init__(self,groups,method="rolling",buffer_bins=2016,age_decay_bins=288):
        if method not in {"frozen","rolling","exponential"}:
            raise ValueError("method must be frozen, rolling, or exponential")
        self.groups=tuple(tuple(group) for group in groups); self.method=method
        self.buffer_bins=buffer_bins; self.age_decay_bins=age_decay_bins
        self.frozen=[]; self.live=[]; self.seen=set()

    @property
    def version(self): return f"received-{self.method}-v1"

    def initialize(self,records):
        records=list(records)
        if not records: raise ValueError("initial calibration archive is empty")
        if len({record.forecast_id for record in records})!=len(records): raise ValueError("duplicate archive residual")
        self.frozen=records.copy(); self.live=records.copy(); self.seen={record.forecast_id for record in records}

    def observe(self,forecast,event,now):
        if event.channel!="feedback" or event.arrival_bin>now or event.observed_bin!=forecast.target_bin or event.sensor_id!=forecast.sensor_id or event.episode_id!=forecast.episode_id:
            raise ValueError("feedback does not causally match forecast")
        if not event.original_valid or forecast.forecast_id in self.seen: return
        self.live.append(Residual(forecast.forecast_id,forecast.sensor_id,forecast.horizon,forecast.target_bin,event.arrival_bin,
                                  abs(event.value-forecast.point)/forecast.scale,forecast.context))
        self.seen.add(forecast.forecast_id)

    def finish_arrival_batch(self,now): pass

    def expire_by_target_time(self,now):
        if self.method!="frozen": self.live=[record for record in self.live if record.target_bin>=now-self.buffer_bins]

    def _records(self,now,sensor,horizon):
        source=self.frozen if self.method=="frozen" else [record for record in self.live if record.arrival_bin<=now and record.target_bin<=now]
        horizon_records=[record for record in source if record.horizon==horizon]
        local=[record for record in horizon_records if record.sensor_id==sensor]
        # Fifty scores is the registered node-archive cutoff; a sparse node uses
        # the same-dataset horizon pool and is reported in diagnostics.
        return (local if len(local)>=50 else horizon_records),len(local)<50

    def predict_one(self,now,sensor,horizon,point,scale,context,missing_fraction):
        records,global_fallback=self._records(now,sensor,horizon)
        if not records: raise ValueError(f"no received residuals for horizon {horizon}")
        scores=np.asarray([record.score for record in records],float)
        intervals=[]
        for level in (.90,.95):
            if self.method=="frozen": quantile=conformal_quantile(scores,1-level)
            elif self.method=="rolling": quantile=float(np.quantile(scores,level,method="higher"))
            else:
                ages=now-np.asarray([record.target_bin for record in records])
                weights=np.exp(-(ages-ages.min())/self.age_decay_bins)
                quantile=weighted_quantile(scores,weights,level)
            width=scale*quantile
            lower=max(0.,point-width); upper=max(lower,point+width)
            intervals.append((level,lower,upper))
        freshest=max(record.target_bin for record in records)
        return tuple(intervals),{"method":self.method,"records":len(records),"global_fallback":global_fallback,
                                "freshest_target_age":max(0,now-freshest),"missing_fraction_ignored":missing_fraction}

    def state_dict(self): return deepcopy(self.__dict__)
    def load_state_dict(self,state): self.__dict__=deepcopy(state)
