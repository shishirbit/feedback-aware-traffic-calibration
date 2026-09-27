from pathlib import Path
import json
import time
import numpy as np
import torch
from ps1.calibration.far_cal import FARCal, Residual
from ps1.data.graphs import local_groups
from ps1.data.prepare import episode_windows
from ps1.models.graph_wavenet_masked import GraphWaveNetConfig, MaskedGraphWaveNet, masked_pinball_loss
from ps1.evaluation.metrics import interval_metrics
from ps1.utils.artifacts import sha256_file, write_json


def load_truth(path):
    data=np.load(path)
    return data["speed_mph"],data["original_valid"]


def issue_contexts(inputs,groups):
    """Observable availability and log mean age from each issue-time history."""
    contexts=np.empty((len(inputs),len(groups),2),dtype=float)
    for origin in range(len(inputs)):
        for sensor,group in enumerate(groups):
            contexts[origin,sensor]=[inputs[origin,:,group,1].mean(),np.log1p((inputs[origin,:,group,2]*288).mean())]
    return contexts


def train_smoke(train_path,adjacency_path,output_dir,epochs=2,seed=11):
    torch.manual_seed(seed); np.random.seed(seed)
    speed,valid=load_truth(train_path)
    medians=np.nanmedian(np.where(valid,speed,np.nan),axis=0)
    mean=float(np.nanmean(np.where(valid,speed,np.nan))); std=max(float(np.nanstd(np.where(valid,speed,np.nan))),1.)
    x,y,mask=episode_windows(speed,valid,medians,mean,std)
    adjacency=np.load(adjacency_path)["adjacency"]
    normalized=adjacency/np.maximum(adjacency.sum(axis=1,keepdims=True),1)
    groups=local_groups(adjacency)
    raw_contexts=issue_contexts(x,groups)
    context_mean=raw_contexts.reshape(-1,2).mean(axis=0)
    context_std=np.maximum(raw_contexts.reshape(-1,2).std(axis=0),1e-8)
    config=GraphWaveNetConfig(num_nodes=speed.shape[1],input_features=7,horizons=12,residual_channels=8,dilation_channels=8,skip_channels=16,end_channels=32,blocks=2,layers=2,dropout=.1)
    model=MaskedGraphWaveNet(config,(torch.tensor(normalized,dtype=torch.float32),))
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    checkpoint=output/"far_gw_smoke.pt"; summary_path=output/"training_summary.json"
    if checkpoint.exists() and summary_path.exists():
        saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
        model.load_state_dict(saved["model"])
        summary=json.loads(summary_path.read_text(encoding="utf-8"))
        return model,dict(mean=saved["mean"],std=saved["std"],medians=np.asarray(saved["medians"]),
                          context_mean=np.asarray(saved["context_mean"]),context_std=np.asarray(saved["context_std"])),summary
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    losses=[]; started=time.perf_counter()
    for _ in range(epochs):
        model.train()
        for start in range(0,len(x),4):
            xb=torch.tensor(x[start:start+4]); yb=torch.tensor(y[start:start+4]); mb=torch.tensor(mask[start:start+4])
            optimizer.zero_grad(); loss=masked_pinball_loss(model(xb),yb,mb)
            if loss is None: continue
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step(); losses.append(float(loss.detach()))
    torch.save({"model":model.state_dict(),"architecture":model.resolved_architecture(),"mean":mean,"std":std,
                "medians":medians.tolist(),"context_mean":context_mean.tolist(),"context_std":context_std.tolist(),"seed":seed},checkpoint)
    summary={"status":"smoke_only","epochs":epochs,"optimizer_steps":len(losses),"first_loss":losses[0],"last_loss":losses[-1],
             "finite_losses":bool(np.isfinite(losses).all()),"train_seconds":time.perf_counter()-started,
             "checkpoint_sha256":sha256_file(checkpoint),"architecture":model.resolved_architecture()}
    write_json(summary_path,summary)
    return model,dict(mean=mean,std=std,medians=medians,context_mean=context_mean,context_std=context_std),summary


def smoke_calibration_eval(model,stats,calibration_path,test_path,adjacency_path,output_dir):
    adjacency=np.load(adjacency_path)["adjacency"]; groups=local_groups(adjacency)
    archive=[]
    cal_speed,cal_valid=load_truth(calibration_path)
    x,y,masks=episode_windows(cal_speed,cal_valid,stats["medians"],stats["mean"],stats["std"])
    cal_contexts=(issue_contexts(x,groups)-stats["context_mean"])/stats["context_std"]
    model.eval()
    with torch.no_grad():
        low,median,high=model(torch.tensor(x))
    # Context is standardized with smoke-training constants; exact calibration
    # replay tests separately verify online timestamp behavior.
    for origin in range(len(x)):
        for h in range(12):
            for sensor in range(x.shape[2]):
                if masks[origin,h,sensor]:
                    point=float(median[origin,h,sensor]*stats["std"]+stats["mean"])
                    lo=float(low[origin,h,sensor]*stats["std"]+stats["mean"])
                    hi=float(high[origin,h,sensor]*stats["std"]+stats["mean"])
                    scale=max((hi-lo)/2,1.); truth=float(y[origin,h,sensor]*stats["std"]+stats["mean"])
                    target=-(len(x)+12)+origin+h
                    archive.append(Residual(f"cal:{origin}:{h}:{sensor}",sensor,h+1,target,target,abs(truth-point)/scale,tuple(cal_contexts[origin,sensor])))
    calibrator=FARCal(groups); calibrator.initialize(archive)
    test_speed,test_valid=load_truth(test_path)
    tx,ty,tm=episode_windows(test_speed,test_valid,stats["medians"],stats["mean"],stats["std"])
    test_contexts=(issue_contexts(tx,groups)-stats["context_mean"])/stats["context_std"]
    with torch.no_grad(): tlow,tmedian,thigh=model(torch.tensor(tx))
    truth=ty*stats["std"]+stats["mean"]; point=tmedian.numpy()*stats["std"]+stats["mean"]
    lower=np.empty_like(point); upper=np.empty_like(point)
    raw_low=tlow.numpy()*stats["std"]+stats["mean"]; raw_high=thigh.numpy()*stats["std"]+stats["mean"]
    for origin in range(len(tx)):
        for h in range(12):
            for sensor in range(tx.shape[2]):
                scale=max((raw_high[origin,h,sensor]-raw_low[origin,h,sensor])/2,1.)
                unavailable=1-float(tx[origin,:,groups[sensor],1].mean())
                intervals,_=calibrator.predict_one(origin,sensor,h+1,float(point[origin,h,sensor]),float(scale),tuple(test_contexts[origin,sensor]),unavailable)
                lower[origin,h,sensor],upper[origin,h,sensor]=intervals[0][1:]
    metrics=interval_metrics(truth,point,lower,upper,tm,.1)
    metrics.update({"status":"smoke_only","calibration_records":len(archive),"test_windows":len(tx)})
    write_json(Path(output_dir)/"evaluation_summary.json",metrics)
    return metrics
