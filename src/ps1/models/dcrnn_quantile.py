"""PyTorch DCRNN adaptation with PS-1 causal inputs and quantile output."""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DCRNNQuantileConfig:
    num_nodes: int
    input_features: int = 7
    history: int = 12
    horizons: int = 12
    hidden_dim: int = 64
    num_layers: int = 2
    diffusion_steps: int = 2


class _DCGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, supports, diffusion_steps):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.diffusion_steps = diffusion_steps
        for index, support in enumerate(supports):
            self.register_buffer(f"support_{index}", support)
        self.support_count = len(supports)
        width = (input_dim + hidden_dim) * (1 + self.support_count * diffusion_steps)
        self.gates = nn.Linear(width, 2 * hidden_dim)
        self.candidate = nn.Linear(width, hidden_dim)
        nn.init.constant_(self.gates.bias, 1.0)

    def _diffuse(self, values):
        features = [values]
        for index in range(self.support_count):
            support = getattr(self, f"support_{index}")
            previous = values
            current = torch.einsum("nm,bmf->bnf", support, values)
            features.append(current)
            for _ in range(2, self.diffusion_steps + 1):
                following = 2 * torch.einsum("nm,bmf->bnf", support, current) - previous
                features.append(following)
                previous, current = current, following
        return torch.cat(features, dim=-1)

    def forward(self, inputs, state):
        joined = torch.cat((inputs, state), dim=-1)
        reset, update = torch.sigmoid(self.gates(self._diffuse(joined))).chunk(2, dim=-1)
        candidate = torch.tanh(self.candidate(self._diffuse(torch.cat((inputs, reset * state), -1))))
        return update * state + (1 - update) * candidate


class DCRNNQuantile(nn.Module):
    version = "far-dcrnn-v1"
    audited_repository_commit = "602afd9d767d3aa1c9b3eac51710d6aeee12c227"

    def __init__(self, config: DCRNNQuantileConfig, supports):
        super().__init__()
        self.config = config
        c = config
        self.encoder = nn.ModuleList(_DCGRUCell(c.input_features if i == 0 else c.hidden_dim,
            c.hidden_dim, supports, c.diffusion_steps) for i in range(c.num_layers))
        self.decoder = nn.ModuleList(_DCGRUCell(1 if i == 0 else c.hidden_dim,
            c.hidden_dim, supports, c.diffusion_steps) for i in range(c.num_layers))
        self.quantile_projection = nn.Linear(c.hidden_dim, 3)

    def resolved_architecture(self):
        return {**asdict(self.config), "version": self.version,
                "parameters": sum(p.numel() for p in self.parameters()),
                "reference_repository_commit": self.audited_repository_commit,
                "reference_license": "MIT", "filter_type": "dual_random_walk",
                "implementation": "project-pytorch-adaptation"}

    def forward(self, inputs):
        c = self.config
        if inputs.ndim != 4 or tuple(inputs.shape[1:3]) != (c.history, c.num_nodes):
            raise ValueError("expected [batch,history,nodes,features]")
        if inputs.shape[-1] != c.input_features:
            raise ValueError("unexpected feature count")
        batch = inputs.shape[0]
        states = [inputs.new_zeros(batch, c.num_nodes, c.hidden_dim) for _ in self.encoder]
        for step in range(c.history):
            value = inputs[:, step]
            for layer, cell in enumerate(self.encoder):
                states[layer] = cell(value, states[layer]); value = states[layer]
        decoder_input = inputs.new_zeros(batch, c.num_nodes, 1)
        outputs = []
        for _ in range(c.horizons):
            value = decoder_input
            for layer, cell in enumerate(self.decoder):
                states[layer] = cell(value, states[layer]); value = states[layer]
            raw = self.quantile_projection(value)
            median = raw[..., 1]
            outputs.append((median - F.softplus(raw[..., 0]), median,
                            median + F.softplus(raw[..., 2])))
            decoder_input = median.unsqueeze(-1)
        return tuple(torch.stack([output[index] for output in outputs], dim=1) for index in range(3))
