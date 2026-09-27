import unittest

try:
    import torch
except ImportError:
    torch=None

if torch is not None:
    from ps1.models.graph_wavenet_masked import GraphWaveNetConfig, MaskedGraphWaveNet, masked_pinball_loss
    from ps1.models.adapter import TorchForecasterAdapter


@unittest.skipIf(torch is None,"PyTorch is not installed")
class ModelTests(unittest.TestCase):
    def model(self):
        config=GraphWaveNetConfig(num_nodes=3,input_features=5,horizons=4,residual_channels=4,
                                  dilation_channels=4,skip_channels=8,end_channels=16,blocks=1,layers=2,dropout=0)
        adjacency=torch.eye(3)
        return MaskedGraphWaveNet(config,(adjacency,))

    def test_shapes_and_noncrossing(self):
        model=self.model()
        low,median,high=model(torch.randn(2,12,3,5))
        self.assertEqual(low.shape,(2,4,3))
        self.assertTrue(bool((low<=median).all() and (median<=high).all()))
        self.assertEqual(model.resolved_architecture()["source_commit"],"6b162e80c59a1d494809252eca055cff93dc66b1")

    def test_masked_loss_and_empty_batch(self):
        model=self.model(); predictions=model(torch.randn(2,12,3,5)); target=torch.randn(2,4,3)
        valid=torch.zeros_like(target,dtype=torch.bool); valid[0,0,0]=True
        loss=masked_pinball_loss(predictions,target,valid)
        self.assertTrue(bool(torch.isfinite(loss))); loss.backward()
        self.assertIsNone(masked_pinball_loss(predictions,target,torch.zeros_like(valid)))

    def test_adapter_calendar_depends_on_issue_time(self):
        class CaptureModel(torch.nn.Module):
            def eval(self): return self
            def forward(self, value):
                self.value=value
                shape=(1,2,value.shape[2])
                return torch.zeros(shape),torch.zeros(shape),torch.ones(shape)
        model=CaptureModel(); adapter=TorchForecasterAdapter(model,0.,1.)
        history=torch.zeros((12,3)).numpy(); mask=torch.ones((12,3),dtype=torch.bool).numpy(); age=history
        adapter.predict(history,mask,age,11); first=model.value.clone()
        adapter.predict(history,mask,age,12); second=model.value.clone()
        self.assertFalse(torch.equal(first[...,3:],second[...,3:]))


if __name__=="__main__": unittest.main()
