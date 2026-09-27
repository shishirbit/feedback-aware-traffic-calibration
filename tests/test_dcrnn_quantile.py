import numpy as np
import torch

from ps1.models.dcrnn_quantile import DCRNNQuantile, DCRNNQuantileConfig
from ps1.models.graph_wavenet_masked import masked_pinball_loss
from ps1.prediction_cache import load_checkpoint_model


def model(nodes=5):
    support = torch.eye(nodes)
    return DCRNNQuantile(DCRNNQuantileConfig(num_nodes=nodes, hidden_dim=8,
        num_layers=1, diffusion_steps=1), (support, support))


def test_dcrnn_shapes_order_and_gradient():
    network = model(); values = torch.randn(2, 12, 5, 7)
    low, median, high = network(values)
    assert low.shape == median.shape == high.shape == (2, 12, 5)
    assert torch.all(low < median) and torch.all(median < high)
    masked_pinball_loss((low, median, high), torch.randn_like(median),
                        torch.ones_like(median, dtype=torch.bool)).backward()
    assert network.quantile_projection.weight.grad is not None


def test_dcrnn_checkpoint_loader(tmp_path):
    network = model(); checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"architecture": network.resolved_architecture(), "model": network.state_dict()}, checkpoint)
    dataset = type("Dataset", (), {"adjacency": np.eye(5, dtype=np.float32)})()
    restored, _ = load_checkpoint_model(checkpoint, dataset)
    assert isinstance(restored, DCRNNQuantile)
