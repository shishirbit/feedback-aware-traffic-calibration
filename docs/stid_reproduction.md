# STID point-backbone reproduction

## Source audit

- Paper: Shao et al., *Spatial-Temporal Identity: A Simple yet Effective
  Baseline for Multivariate Time Series Forecasting*, CIKM 2022.
- Official repository: `GestaltCogTeam/STID`.
- Audited commit: `e8b313bc591bdd0101a1619962c9b503e75127c0`.
- License: Apache-2.0.

The PS-1 adapter retains STID's flattened-history projection, learned node,
time-of-day, and day-of-week identities, three residual pointwise MLP blocks,
and direct multihorizon projection. It supplies the registered seven causal
input channels and replaces the point head with the same noncrossing
0.05/0.50/0.95 head used by the other frozen-backbone calibration audits.

## Registered adaptation

- History and horizon: 12 five-minute bins each.
- Embeddings: 32 dimensions each for history, node, time of day, and day of week.
- Three residual MLP blocks with dropout 0.15.
- Adam, learning rate 0.002, weight decay 0.0001, batch size 64.
- Maximum 100 epochs and patience 30 on the tuning split.
- Seeds 11, 22, and 33 for METR-LA and PEMS-BAY.

GPU feasibility succeeded in the existing Python 3.11 / PyTorch 2.13.0+cu130
environment: batch 64 used 122,500 parameters on METR-LA and 126,276 on
PEMS-BAY. All six trainings and the complete 114-stage C3/C4 evaluation matrix
finished successfully.

| Dataset / condition | Rolling − FAR-Cal (95% CI) | ACI − FAR-Cal (95% CI) | Coverage | Rule |
|---|---:|---:|---:|---:|
| METR-LA C3 | 1.5509 [0.7963, 2.4513] | 1.5295 [0.7781, 2.4189] | 0.8891 | Met |
| METR-LA C4 | 1.8671 [1.0134, 2.9236] | 1.9169 [1.0804, 2.9395] | 0.8880 | Met |
| PEMS-BAY C3 | -0.0226 [-0.1687, 0.1666] | -0.0107 [-0.1555, 0.1799] | 0.9002 | Not met |
| PEMS-BAY C4 | 0.0673 [-0.1183, 0.2755] | 0.0754 [-0.1217, 0.2974] | 0.9062 | Not met |

STID replicates the METR-LA result but not the PEMS-BAY result. Combined with
FAR-GW and STAEformer, this supports multi-backbone transfer on METR-LA while
rejecting universal backbone-independent superiority.
