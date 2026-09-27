import torch

from ps1.models.graph_wavenet_masked import masked_pinball_loss
from ps1.models.staeformer_quantile import STAEformerQuantile, STAEformerQuantileConfig


def test_staeformer_quantile_shapes_and_order():
    config = STAEformerQuantileConfig(num_nodes=5, input_embedding_dim=8,
        tod_embedding_dim=4, dow_embedding_dim=4, adaptive_embedding_dim=8,
        feed_forward_dim=32, num_heads=4, num_layers=1)
    model = STAEformerQuantile(config)
    values = torch.randn(2, 12, 5, 7)
    low, median, high = model(values)
    assert low.shape == median.shape == high.shape == (2, 12, 5)
    assert torch.all(low < median) and torch.all(median < high)
    target = torch.randn_like(median); valid = torch.ones_like(target, dtype=torch.bool)
    masked_pinball_loss((low, median, high), target, valid).backward()
    assert model.adaptive_embedding.grad is not None


def test_staeformer_rejects_wrong_shape():
    model = STAEformerQuantile(STAEformerQuantileConfig(num_nodes=5))
    try:
        model(torch.randn(2, 11, 5, 7))
    except ValueError as error:
        assert "expected" in str(error)
    else:
        raise AssertionError("wrong history length accepted")
