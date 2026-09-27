import numpy as np


def causal_features(speed_mph, valid, sensor_medians, bins_per_day=288):
    speed=np.asarray(speed_mph,float); valid=np.asarray(valid,bool)
    medians=np.asarray(sensor_medians,float)
    if speed.shape!=valid.shape or speed.ndim!=2 or speed.shape[1]!=len(medians):
        raise ValueError("episode arrays or medians do not align")
    filled=np.empty_like(speed); age=np.empty_like(speed)
    last_value=medians.copy(); last_bin=np.full(len(medians),-1,dtype=int)
    for now in range(len(speed)):
        observed=valid[now]&np.isfinite(speed[now])
        last_value[observed]=speed[now,observed]; last_bin[observed]=now
        filled[now]=last_value
        age[now]=np.where(last_bin>=0,now-last_bin,now+1)
    phase=2*np.pi*np.arange(len(speed))/bins_per_day
    day_phase=2*np.pi*np.floor(np.arange(len(speed))/bins_per_day)/7
    temporal=np.stack([np.sin(phase),np.cos(phase),np.sin(day_phase),np.cos(day_phase)],axis=1)
    temporal=np.broadcast_to(temporal[:,None,:],(len(speed),speed.shape[1],4))
    return filled,age,temporal


def episode_windows(speed_mph, valid, sensor_medians, mean, std, history=12, horizon=12):
    filled,age,temporal=causal_features(speed_mph,valid,sensor_medians)
    inputs=[]; targets=[]; masks=[]
    for issue in range(history-1,len(filled)-horizon):
        rows=slice(issue-history+1,issue+1)
        feature=np.concatenate([((filled[rows]-mean)/std)[...,None],valid[rows][...,None].astype(float),
                                np.minimum(age[rows],288)[...,None]/288,temporal[rows]],axis=-1)
        inputs.append(feature); targets.append((speed_mph[issue+1:issue+horizon+1]-mean)/std); masks.append(valid[issue+1:issue+horizon+1])
    return np.asarray(inputs,np.float32),np.asarray(targets,np.float32),np.asarray(masks,bool)
