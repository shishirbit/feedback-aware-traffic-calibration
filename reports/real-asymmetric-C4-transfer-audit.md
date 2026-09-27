# Real-data asymmetric FAR-Cal C4 transfer audit

The C3-locked asymmetric design was evaluated under C4 permanent feedback loss across three model seeds and three fault seeds. The resource-bounded protocol uses hourly fault-focused origins and paired one-day moving-block bootstraps stratified by fault seed. Positive differences favor FAR-Cal.

| Dataset | FAR-Cal score | Coverage | Rolling − FAR-Cal (95% CI) | ACI − FAR-Cal (95% CI) | Rule met |
|---|---:|---:|---:|---:|---:|
| PEMS-BAY | 10.5644 | 0.9068 | -0.2651 [-0.4350, -0.0955] | -0.2908 [-0.4839, -0.1051] | No |
| METR-LA | 23.4332 | 0.8958 | 0.9507 [0.5331, 1.4261] | 0.9512 [0.5656, 1.3808] | Yes |

Asymmetric FAR-Cal is superior on METR-LA but significantly worse than both comparators on PEMS-BAY. The real-data C4 evidence therefore rejects broad cross-dataset superiority.

Machine-readable results: `artifacts/evaluation/pems_bay-asymmetric-C4-transfer-audit.json` and `artifacts/evaluation/metr_la-asymmetric-C4-transfer-audit.json`.
