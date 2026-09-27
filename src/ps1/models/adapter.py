import numpy as np
import torch


class TorchForecasterAdapter:
    version="far-gw-v1"

    def __init__(self,model,mean,std,device="cpu",start_timestamp_ns=None):
        self.model,self.mean,self.std,self.device=model,float(mean),float(std),torch.device(device)
        self.start_timestamp_ns=None if start_timestamp_ns is None else int(start_timestamp_ns)

    def predict(self,history,mask,age,issue_bin=None):
        length,nodes=history.shape
        issue_bin=length-1 if issue_bin is None else int(issue_bin)
        bins=np.arange(issue_bin-length+1,issue_bin+1)
        if self.start_timestamp_ns is None:
            tod=2*np.pi*(bins%288)/288; dow=2*np.pi*((bins//288)%7)/7
        else:
            stamps=(np.datetime64(self.start_timestamp_ns,"ns")+bins*np.timedelta64(5,"m"))
            day=stamps.astype("datetime64[D]")
            minute=(stamps-day).astype("timedelta64[m]").astype(np.int64)
            weekday=(day.astype(np.int64)+3)%7
            tod=2*np.pi*minute/1440.; dow=2*np.pi*weekday/7
        temporal=np.stack([np.sin(tod),np.cos(tod),np.sin(dow),np.cos(dow)],axis=1)
        temporal=np.broadcast_to(temporal[:,None,:],(length,nodes,4))
        feature=np.concatenate([((history-self.mean)/self.std)[...,None],mask[...,None].astype(float),age[...,None],temporal],axis=-1)
        self.model.eval()
        with torch.no_grad():
            low,median,high=self.model(torch.as_tensor(feature[None],dtype=torch.float32,device=self.device))
        low,median,high=(value[0].cpu().numpy()*self.std+self.mean for value in (low,median,high))
        scale=np.maximum((high-low)/2,1.)
        return median,scale,(low,median,high)
