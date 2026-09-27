"""Consumer engine receives released events only; has no truth/fault reference."""
import pickle
from .ledger import Forecast, ForecastLedger


class OnlineEngine:
    def __init__(self, history, model, calibrator, context_mean=(0.,0.), context_std=(1.,1.)):
        import numpy as np
        self.history, self.model, self.calibrator = history, model, calibrator
        self.context_mean = np.asarray(context_mean, float)
        self.context_std = np.asarray(context_std, float)
        if (self.context_std <= 0).any():
            raise ValueError("positive training-only context standard deviations required")
        self.ledger = ForecastLedger()
        self.last_bin = None
        self.diagnostics = []

    def step(self, now, released_events, issue=True):
        import numpy as np
        if self.last_bin is not None and now <= self.last_bin:
            raise ValueError("engine boundaries must increase")
        events = sorted(released_events, key=lambda e:(e.arrival_bin,e.observed_bin,e.sensor_id,e.channel,e.event_id))
        for event in events:
            if event.arrival_bin > now:
                raise ValueError("future release exposed to engine")
            if event.channel == "input":
                self.history.ingest(event,now)
            else:
                self.ledger.ingest_feedback(event,self.calibrator,now)
        self.calibrator.finish_arrival_batch(now)
        self.calibrator.expire_by_target_time(now)
        if issue:
            x,mask,age = self.history.causal_window(now)
            point,scale,_ = self.model.predict(x,mask,np.minimum(age,288)/288,now)
            for sensor,group in enumerate(self.calibrator.groups):
                available = float(mask[:,group].mean())
                raw = np.array([available,np.log1p(age[:,group].mean())])
                context = tuple((raw-self.context_mean)/self.context_std)
                for h in range(1,len(point)+1):
                    intervals,diagnostics = self.calibrator.predict_one(now,sensor,h,point[h-1,sensor],scale[h-1,sensor],context,1-available)
                    fid=f"{self.history.episode_id}:{now}:{sensor}:{h}"
                    forecast=Forecast(fid,self.history.episode_id,now,now+h,h,sensor,float(point[h-1,sensor]),float(scale[h-1,sensor]),intervals,context,self.model.version,self.calibrator.version)
                    self.ledger.append(forecast)
                    self.diagnostics.append((fid,diagnostics))
        self.last_bin=now

    def checkpoint(self):
        """Trusted local checkpoints only: never unpickle external artifacts."""
        return pickle.dumps(self,protocol=5)

    @classmethod
    def restore(cls, blob):
        engine=pickle.loads(blob)
        if not isinstance(engine,cls):
            raise ValueError("wrong checkpoint object")
        return engine
