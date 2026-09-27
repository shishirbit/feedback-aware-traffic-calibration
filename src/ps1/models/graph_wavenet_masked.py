"""Mask/age Graph WaveNet adaptation with a noncrossing quantile head.

Backbone design is adapted from Zonghan Wu et al.'s author implementation,
commit 6b162e80c59a1d494809252eca055cff93dc66b1 (MIT). Changes include a
batch/time/node/feature interface, explicit mask/age channels, registered
architecture serialization, and a three-quantile head.
"""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GraphWaveNetConfig:
    num_nodes: int
    input_features: int = 5
    horizons: int = 12
    residual_channels: int = 32
    dilation_channels: int = 32
    skip_channels: int = 256
    end_channels: int = 512
    blocks: int = 4
    layers: int = 2
    kernel_size: int = 2
    graph_order: int = 2
    adaptive_rank: int = 10
    dropout: float = .3


class NeighborhoodConvolution(nn.Module):
    def forward(self, values, adjacency):
        return torch.einsum("bcnt,nm->bcmt", values, adjacency).contiguous()


class DiffusionGraphConvolution(nn.Module):
    def __init__(self, channels_in, channels_out, support_count, order, dropout):
        super().__init__()
        self.neighborhood=NeighborhoodConvolution()
        self.order=order
        self.project=nn.Conv2d((order*support_count+1)*channels_in,channels_out,1)
        self.dropout=dropout

    def forward(self, values, supports):
        outputs=[values]
        for adjacency in supports:
            current=self.neighborhood(values,adjacency); outputs.append(current)
            for _ in range(2,self.order+1):
                current=self.neighborhood(current,adjacency); outputs.append(current)
        return F.dropout(self.project(torch.cat(outputs,dim=1)),self.dropout,self.training)


class MaskedGraphWaveNet(nn.Module):
    version="far-gw-v1"

    def __init__(self, config, fixed_supports=()):
        super().__init__()
        self.config=config
        self.register_buffer("fixed_supports",torch.stack(tuple(fixed_supports)) if fixed_supports else torch.empty(0,config.num_nodes,config.num_nodes))
        rank=min(config.adaptive_rank,config.num_nodes)
        self.node_source=nn.Parameter(torch.randn(config.num_nodes,rank))
        self.node_target=nn.Parameter(torch.randn(rank,config.num_nodes))
        support_count=len(fixed_supports)+1
        c=config
        self.start=nn.Conv2d(c.input_features,c.residual_channels,1)
        self.filters,self.gates,self.residuals,self.skips,self.norms,self.graphs=(nn.ModuleList() for _ in range(6))
        receptive=1
        for _ in range(c.blocks):
            dilation=1
            for _ in range(c.layers):
                self.filters.append(nn.Conv2d(c.residual_channels,c.dilation_channels,(1,c.kernel_size),dilation=(1,dilation)))
                self.gates.append(nn.Conv2d(c.residual_channels,c.dilation_channels,(1,c.kernel_size),dilation=(1,dilation)))
                self.residuals.append(nn.Conv2d(c.dilation_channels,c.residual_channels,1))
                self.skips.append(nn.Conv2d(c.dilation_channels,c.skip_channels,1))
                self.norms.append(nn.BatchNorm2d(c.residual_channels))
                self.graphs.append(DiffusionGraphConvolution(c.dilation_channels,c.residual_channels,support_count,c.graph_order,c.dropout))
                receptive+=(c.kernel_size-1)*dilation
                dilation*=2
        self.receptive_field=receptive
        self.end1=nn.Conv2d(c.skip_channels,c.end_channels,1)
        self.end2=nn.Conv2d(c.end_channels,3*c.horizons,1)

    def resolved_architecture(self):
        return {**asdict(self.config),"receptive_field":self.receptive_field,
                "parameters":sum(parameter.numel() for parameter in self.parameters()),
                "source_commit":"6b162e80c59a1d494809252eca055cff93dc66b1"}

    def forward(self, inputs):
        if inputs.ndim!=4 or inputs.shape[2]!=self.config.num_nodes or inputs.shape[3]!=self.config.input_features:
            raise ValueError("expected [batch,time,nodes,features]")
        x=inputs.permute(0,3,2,1)
        if x.shape[-1]<self.receptive_field:
            x=F.pad(x,(self.receptive_field-x.shape[-1],0,0,0))
        x=self.start(x)
        adaptive=F.softmax(F.relu(self.node_source@self.node_target),dim=1)
        supports=[*self.fixed_supports.unbind(0),adaptive]
        skip=None
        for filter_conv,gate_conv,residual_conv,skip_conv,norm,graph in zip(self.filters,self.gates,self.residuals,self.skips,self.norms,self.graphs):
            residual=x
            x=torch.tanh(filter_conv(x))*torch.sigmoid(gate_conv(x))
            contribution=skip_conv(x)
            skip=contribution if skip is None else skip[:,:,:,-contribution.shape[-1]:]+contribution
            x=graph(x,supports)+residual[:,:,:,-x.shape[-1]:]
            x=norm(x)
        raw=self.end2(F.relu(self.end1(F.relu(skip))))[:,:,:,-1]
        raw=raw.reshape(raw.shape[0],3,self.config.horizons,self.config.num_nodes)
        median=raw[:,1]
        low=median-F.softplus(raw[:,0])
        high=median+F.softplus(raw[:,2])
        return low,median,high


def masked_pinball_loss(predictions, target, valid, quantiles=(.05,.5,.95)):
    if not bool(valid.any()):
        return None
    losses=[]
    for prediction,quantile in zip(predictions,quantiles):
        error=target-prediction
        losses.append(torch.maximum(quantile*error,(quantile-1)*error)[valid].mean())
    return torch.stack(losses).mean()
