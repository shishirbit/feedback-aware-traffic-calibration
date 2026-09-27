"""STID adapter with PS-1 causal inputs and a noncrossing quantile head."""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class STIDQuantileConfig:
    num_nodes: int
    input_features: int = 7
    history: int = 12
    horizons: int = 12
    embedding_dim: int = 32
    node_dim: int = 32
    tod_dim: int = 32
    dow_dim: int = 32
    num_layers: int = 3
    dropout: float = .15
    steps_per_day: int = 288


class _ResidualMLP(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.first = nn.Conv2d(width, width, 1)
        self.second = nn.Conv2d(width, width, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        return values + self.second(self.dropout(F.relu(self.first(values))))


class STIDQuantile(nn.Module):
    version = "far-stid-v1"
    audited_repository_commit = "e8b313bc591bdd0101a1619962c9b503e75127c0"

    def __init__(self, config: STIDQuantileConfig):
        super().__init__()
        self.config = config
        c = config
        self.series_projection = nn.Conv2d(c.input_features * c.history, c.embedding_dim, 1)
        self.node_embedding = nn.Parameter(torch.empty(c.num_nodes, c.node_dim))
        self.tod_embedding = nn.Parameter(torch.empty(c.steps_per_day, c.tod_dim))
        self.dow_embedding = nn.Parameter(torch.empty(7, c.dow_dim))
        for value in (self.node_embedding, self.tod_embedding, self.dow_embedding):
            nn.init.xavier_uniform_(value)
        width = c.embedding_dim + c.node_dim + c.tod_dim + c.dow_dim
        self.encoder = nn.Sequential(*(_ResidualMLP(width, c.dropout) for _ in range(c.num_layers)))
        self.output_projection = nn.Conv2d(width, 3 * c.horizons, 1)

    @staticmethod
    def _cyclic_index(sine, cosine, bins):
        angle = torch.atan2(sine, cosine).remainder(2 * torch.pi)
        return torch.floor(angle * bins / (2 * torch.pi)).long().clamp_(0, bins - 1)

    def resolved_architecture(self):
        return {**asdict(self.config), "version": self.version,
                "parameters": sum(p.numel() for p in self.parameters()),
                "reference_repository_commit": self.audited_repository_commit,
                "reference_license": "Apache-2.0", "implementation": "project-adaptation"}

    def forward(self, inputs):
        c = self.config
        if inputs.ndim != 4 or tuple(inputs.shape[1:3]) != (c.history, c.num_nodes):
            raise ValueError("expected [batch,history,nodes,features]")
        if inputs.shape[-1] != c.input_features:
            raise ValueError("unexpected feature count")
        batch = inputs.shape[0]
        series = inputs.permute(0, 2, 1, 3).reshape(batch, c.num_nodes, -1)
        series = self.series_projection(series.transpose(1, 2).unsqueeze(-1))
        tod = self._cyclic_index(inputs[:, -1, :, 3], inputs[:, -1, :, 4], c.steps_per_day)
        dow = self._cyclic_index(inputs[:, -1, :, 5], inputs[:, -1, :, 6], 7)
        node = self.node_embedding.T.unsqueeze(0).unsqueeze(-1).expand(batch, -1, -1, -1)
        tod_embedding = self.tod_embedding[tod].permute(0, 2, 1).unsqueeze(-1)
        dow_embedding = self.dow_embedding[dow].permute(0, 2, 1).unsqueeze(-1)
        raw = self.output_projection(self.encoder(torch.cat((series, node, tod_embedding, dow_embedding), 1)))
        raw = raw.squeeze(-1).view(batch, 3, c.horizons, c.num_nodes)
        median = raw[:, 1]
        return median - F.softplus(raw[:, 0]), median, median + F.softplus(raw[:, 2])
