# CoRel reproduction audit

Checked 2026-09-16 against the official MIT-licensed repository at commit
`4504c4edf76128bfe657762a41ec9eb043038ca2`.

The source trains a graph quantile network on lagged prediction residuals. Its
METR-LA protocol uses a 12-bin residual window, predicts all 12 residual horizons
as output features, learns up to 20 neighbors, and emits 39 quantiles. The paper
uses a 40/40/20 train/calibration/test split and tunes on the first quarter of its
calibration portion. Its optional test-time adaptation periodically fine-tunes
node embeddings from the latest observed residual block.

This protocol is reproducible but is not directly interchangeable with PS-1.
PS-1 uses fixed 60/10/10/20 boundaries and separately controls the arrival of
each forecast outcome. A project adapter must therefore train the unchanged
CoRel architecture only on PS-1 calibration residuals, expose horizons 3/6/12,
and release residual inputs according to the same feedback manifest as every
other calibrator. That result must be labeled a multihorizon feedback adaptation;
a source-protocol run should be reported separately.

An isolated Python 3.11 environment successfully imported the official encoder
and completed a forward/backward architecture smoke test with output shape
`[3 quantiles, 2 samples, 1 step, 20 nodes, 12 horizons]` and 76,532 parameters.
The current PyTorch 2.14 CPU wheel has no matching Windows PyG extension wheels,
so this environment uses PyTorch 2.5.1 CPU plus the official PyG 2.5 CPU wheels.
Resolved packages are recorded in `requirements-corel.lock`. This establishes
source compatibility only; it is not comparator evidence.

Full PS-1 adapter training completed on all three METR-LA model seeds. Early
stopping selected epochs 5, 10, and 20, with standardized validation pinball
losses 0.132512, 0.130697, and 0.132488. Each run used 3,063 examples and held
out 341 calibration examples for validation; CPU runtimes were 2,643.5, 3,274.9,
and 4,589.1 seconds. These are trained calibrator checkpoints; test-set comparator
evidence still requires the feedback-arrival evaluation adapter.

PEMS-BAY seed 11 also completed after a checkpointed system restart. It selected
epoch 17, stopped after epoch 32, and achieved standardized validation pinball
loss 0.226358. The two invocations took 9,793.2 CPU seconds in total.
PEMS-BAY seed 22 selected epoch 19, stopped after epoch 34, achieved validation
pinball loss 0.225812, and took 10,230.7 CPU seconds.
PEMS-BAY seed 33 selected epoch 18, stopped after epoch 33, achieved validation
pinball loss 0.225816, and took 9,894.9 CPU seconds. All six registered CoRel
adapter trainings are therefore complete.

The subsequent 36-cell C3/C4 evaluation is complete. FAR-Cal achieved lower
interval score than CoRel in all four dataset-condition summaries, with paired
CoRel-minus-FAR-Cal effects from 1.6960 to 2.2547 and every 95% lower bound above
zero. See `reports/corel-transfer-audit.md` for the complete claim boundary.

Primary sources:

- https://proceedings.mlr.press/v267/cini25a.html
- https://github.com/andreacini/corel/tree/4504c4edf76128bfe657762a41ec9eb043038ca2
