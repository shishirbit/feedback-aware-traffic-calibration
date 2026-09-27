"""STAEformer-style factorized Transformer with a noncrossing quantile head.

This is an independent implementation from the architecture described by Liu
et al. (CIKM 2023).  The authors' repository was audited at the commit recorded
in ``resolved_architecture`` but has no license file, so no source code is
copied from that repository.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class STAEformerQuantileConfig:
    num_nodes: int
    input_features: int = 7
    history: int = 12
    horizons: int = 12
    input_embedding_dim: int = 24
    tod_embedding_dim: int = 24
    dow_embedding_dim: int = 24
    adaptive_embedding_dim: int = 80
    feed_forward_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = .1
    steps_per_day: int = 288


class _AxisEncoder(nn.Module):
    def __init__(self, width, heads, feed_forward, dropout):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=feed_forward,
            dropout=dropout, activation="relu", batch_first=True, norm_first=False,
        )

    def forward(self, values, axis):
        # [B,T,N,D], with attention independently over the selected axis.
        if axis == 1:
            batch, steps, nodes, width = values.shape
            encoded = self.block(values.permute(0, 2, 1, 3).reshape(batch * nodes, steps, width))
            return encoded.reshape(batch, nodes, steps, width).permute(0, 2, 1, 3)
        if axis == 2:
            batch, steps, nodes, width = values.shape
            encoded = self.block(values.reshape(batch * steps, nodes, width))
            return encoded.reshape(batch, steps, nodes, width)
        raise ValueError("axis must be temporal (1) or spatial (2)")


class STAEformerQuantile(nn.Module):
    version = "far-staef-v1"
    audited_repository_commit = "fc49d39b2f1a8e3cf37b6289d7240680e1690f3f"

    def __init__(self, config: STAEformerQuantileConfig):
        super().__init__()
        self.config = config
        c = config
        self.input_projection = nn.Linear(c.input_features, c.input_embedding_dim)
        self.tod_embedding = nn.Embedding(c.steps_per_day, c.tod_embedding_dim)
        self.dow_embedding = nn.Embedding(7, c.dow_embedding_dim)
        self.adaptive_embedding = nn.Parameter(
            torch.empty(c.history, c.num_nodes, c.adaptive_embedding_dim)
        )
        nn.init.xavier_uniform_(self.adaptive_embedding)
        width = c.input_embedding_dim + c.tod_embedding_dim + c.dow_embedding_dim + c.adaptive_embedding_dim
        if width % c.num_heads:
            raise ValueError("model width must be divisible by num_heads")
        self.temporal = nn.ModuleList(
            _AxisEncoder(width, c.num_heads, c.feed_forward_dim, c.dropout)
            for _ in range(c.num_layers)
        )
        self.spatial = nn.ModuleList(
            _AxisEncoder(width, c.num_heads, c.feed_forward_dim, c.dropout)
            for _ in range(c.num_layers)
        )
        self.output_projection = nn.Linear(c.history * width, 3 * c.horizons)

    def resolved_architecture(self):
        return {
            **asdict(self.config), "version": self.version,
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "reference_repository_commit": self.audited_repository_commit,
            "implementation": "independent-paper-based",
        }

    @staticmethod
    def _cyclic_index(sine, cosine, bins):
        angle = torch.atan2(sine, cosine).remainder(2 * torch.pi)
        return torch.floor(angle * bins / (2 * torch.pi)).long().clamp_(0, bins - 1)

    def forward(self, inputs):
        c = self.config
        if inputs.ndim != 4 or tuple(inputs.shape[1:3]) != (c.history, c.num_nodes):
            raise ValueError("expected [batch,history,nodes,features]")
        if inputs.shape[-1] != c.input_features:
            raise ValueError("unexpected feature count")
        # Dataset channels 3:7 are sin/cos time of day and day of week.
        tod = self._cyclic_index(inputs[..., 3], inputs[..., 4], c.steps_per_day)
        dow = self._cyclic_index(inputs[..., 5], inputs[..., 6], 7)
        batch = inputs.shape[0]
        adaptive = self.adaptive_embedding.unsqueeze(0).expand(batch, -1, -1, -1)
        encoded = torch.cat((self.input_projection(inputs), self.tod_embedding(tod),
                             self.dow_embedding(dow), adaptive), dim=-1)
        for block in self.temporal:
            encoded = block(encoded, 1)
        for block in self.spatial:
            encoded = block(encoded, 2)
        raw = self.output_projection(encoded.permute(0, 2, 1, 3).flatten(2))
        raw = raw.view(batch, c.num_nodes, 3, c.horizons).permute(0, 2, 3, 1)
        median = raw[:, 1]
        return median - F.softplus(raw[:, 0]), median, median + F.softplus(raw[:, 2])
