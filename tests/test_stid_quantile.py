import torch

from ps1.models.graph_wavenet_masked import masked_pinball_loss
from ps1.models.stid_quantile import STIDQuantile, STIDQuantileConfig
from ps1.prediction_cache import load_checkpoint_model


def test_stid_quantile_shapes_order_and_gradient():
    model = STIDQuantile(STIDQuantileConfig(num_nodes=5, embedding_dim=8,
        node_dim=8, tod_dim=8, dow_dim=8, num_layers=1))
    values = torch.randn(2, 12, 5, 7)
    low, median, high = model(values)
    assert low.shape == median.shape == high.shape == (2, 12, 5)
    assert torch.all(low < median) and torch.all(median < high)
    target = torch.randn_like(median); valid = torch.ones_like(target, dtype=torch.bool)
    masked_pinball_loss((low, median, high), target, valid).backward()
    assert model.node_embedding.grad is not None


def test_stid_rejects_wrong_shape():
    model = STIDQuantile(STIDQuantileConfig(num_nodes=5))
    try:
        model(torch.randn(2, 11, 5, 7))
    except ValueError as error:
        assert "expected" in str(error)
    else:
        raise AssertionError("wrong history length accepted")


def test_stid_checkpoint_loader(tmp_path):
    model = STIDQuantile(STIDQuantileConfig(num_nodes=5, embedding_dim=8,
        node_dim=8, tod_dim=8, dow_dim=8, num_layers=1))
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"architecture": model.resolved_architecture(), "model": model.state_dict()}, checkpoint)
    restored, saved = load_checkpoint_model(checkpoint, None)
    assert isinstance(restored, STIDQuantile)
    assert saved["architecture"]["version"] == STIDQuantile.version
