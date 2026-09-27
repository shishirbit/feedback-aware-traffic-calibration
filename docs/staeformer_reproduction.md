# STAEformer second-backbone audit

## Source and license decision

- Paper: Liu et al., *Spatio-Temporal Adaptive Embedding Makes Vanilla
  Transformer SOTA for Traffic Forecasting*, CIKM 2023.
- Official repository: `XDZhelheim/STAEformer`.
- Audited commit: `fc49d39b2f1a8e3cf37b6289d7240680e1690f3f`
  (2025-11-15).
- The audited repository contains no license file. Its code is retained only in
  the ignored `third_party/staeformer` audit checkout and is not redistributed.

The project implementation in `src/ps1/models/staeformer_quantile.py` was
written independently from the published architecture description. It uses
factorized temporal and spatial Transformer encoders, learned time-of-day and
day-of-week embeddings, a learned spatiotemporal adaptive embedding, and the
paper's mixed time/feature output projection. PS-1 adds its registered seven
causal input channels and a noncrossing 0.05/0.50/0.95 quantile head.

## Registered adaptation

- History and horizon: 12 five-minute bins each.
- Width: 24 input + 24 time-of-day + 24 day-of-week + 80 adaptive = 152.
- Three temporal and three spatial blocks; four heads; feed-forward width 256.
- Batch size 32 after GPU feasibility checks (64 approached the 8 GB limit and
  128 failed); Adam, learning rate 0.001, weight decay 0.0003.
- Early stopping: maximum 200 epochs, patience 30, tuning partition only.
- Seeds: 11, 22, and 33 for both METR-LA and PEMS-BAY.
- Training uses the same split-safe windows, causal fill, mask, age, calendar
  features, missingness augmentation, target scaling, and pinball loss as FAR-GW.

## Runtime environment

The isolated `.venv-staeformer` environment uses Python 3.11 and PyTorch
2.13.0+cu130. `requirements-staeformer.lock` records all packages. The NVIDIA
GeForce RTX 5060 Laptop GPU was detected successfully. A METR-LA batch-32
feasibility step used the 1,302,876-parameter model and completed without an
out-of-memory error. The earlier CPU batch-16 check took 6.81 seconds per step,
so GPU execution is required for the registered matrix.

All six checkpoints and the 114-stage evaluation matrix completed. The runner
created clean calibration caches, all 36 C3/C4 faulted caches, matching rolling
and ACI runs, asymmetric FAR-Cal runs, and four dataset/scenario aggregate audits.

| Dataset / condition | Rolling − FAR-Cal (95% CI) | ACI − FAR-Cal (95% CI) | Coverage | Rule |
|---|---:|---:|---:|---:|
| METR-LA C3 | 1.4972 [0.7436, 2.3621] | 1.5022 [0.7597, 2.3411] | 0.8965 | Met |
| METR-LA C4 | 1.7914 [0.9847, 2.7599] | 1.8495 [1.0384, 2.7850] | 0.8916 | Met |
| PEMS-BAY C3 | 0.2531 [0.0849, 0.4478] | 0.2616 [0.0978, 0.4520] | 0.8945 | Met |
| PEMS-BAY C4 | 0.2598 [0.0983, 0.4390] | 0.2743 [0.1007, 0.4675] | 0.9101 | Met |

## Claim boundary

The result demonstrates successful transfer to a second tested frozen backbone.
It does not establish universal backbone-independent superiority: FAR-GW failed
the registered rule on PEMS-BAY C3 and C4, so the PEMS-BAY conclusion depends on
the frozen backbone. See `reports/cross-backbone-transfer-synthesis.md`.
