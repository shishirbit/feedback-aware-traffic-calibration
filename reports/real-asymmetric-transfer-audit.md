# Real-data asymmetric FAR-Cal transfer audit

The SUMO-locked asymmetric design was transferred without outcome-driven retuning to an hourly, fault-focused C3 audit. The scalable variant uses causal age/context-weighted signed residuals from each sensor's two-neighbor group, capped at 256 records per sensor. Confidence intervals use paired one-day moving-block bootstraps. Positive differences favor FAR-Cal.

| Dataset | Model × fault seeds | FAR-Cal score | Coverage | Rolling − FAR-Cal (95% CI) | ACI − FAR-Cal (95% CI) | Rule met |
|---|---:|---:|---:|---:|---:|---:|
| PEMS-BAY | 3 × 3 | 10.3271 | 0.9006 | -0.0786 [-0.2260, 0.1046] | -0.0936 [-0.2487, 0.0951] | No |
| METR-LA | 3 × 3 | 20.3370 | 0.8985 | 0.7932 [0.4293, 1.2330] | 0.7723 [0.4141, 1.2039] | Yes |

The complete 3 × 3 transfer audit supports asymmetric FAR-Cal on METR-LA but does not establish superiority on PEMS-BAY. Therefore it does not satisfy a broad cross-dataset superiority claim.

Machine-readable results: `artifacts/evaluation/pems_bay-asymmetric-transfer-audit.json` and `artifacts/evaluation/metr_la-asymmetric-transfer-audit.json`.
