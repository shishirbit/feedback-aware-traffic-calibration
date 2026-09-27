# Cross-backbone FAR-Cal transfer synthesis

| Frozen backbone | Dataset | Condition | Rolling − FAR-Cal (95% CI) | ACI − FAR-Cal (95% CI) | Coverage | Rule |
|---|---|---|---:|---:|---:|---:|
| FAR-GW | METR-LA | C3 | 0.7932 [0.4293, 1.2330] | 0.7723 [0.4141, 1.2039] | 0.8985 | Met |
| FAR-GW | METR-LA | C4 | 0.9507 [0.5331, 1.4261] | 0.9512 [0.5656, 1.3808] | 0.8958 | Met |
| FAR-GW | PEMS-BAY | C3 | -0.0786 [-0.2260, 0.1046] | -0.0936 [-0.2487, 0.0951] | 0.9006 | Not met |
| FAR-GW | PEMS-BAY | C4 | -0.2651 [-0.4350, -0.0955] | -0.2908 [-0.4839, -0.1051] | 0.9068 | Not met |
| STAEformer | METR-LA | C3 | 1.4972 [0.7436, 2.3621] | 1.5022 [0.7597, 2.3411] | 0.8965 | Met |
| STAEformer | METR-LA | C4 | 1.7914 [0.9847, 2.7599] | 1.8495 [1.0384, 2.7850] | 0.8916 | Met |
| STAEformer | PEMS-BAY | C3 | 0.2531 [0.0849, 0.4478] | 0.2616 [0.0978, 0.4520] | 0.8945 | Met |
| STAEformer | PEMS-BAY | C4 | 0.2598 [0.0983, 0.4390] | 0.2743 [0.1007, 0.4675] | 0.9101 | Met |
| STID | METR-LA | C3 | 1.5509 [0.7963, 2.4513] | 1.5295 [0.7781, 2.4189] | 0.8891 | Met |
| STID | METR-LA | C4 | 1.8671 [1.0134, 2.9236] | 1.9169 [1.0804, 2.9395] | 0.8880 | Met |
| STID | PEMS-BAY | C3 | -0.0226 [-0.1687, 0.1666] | -0.0107 [-0.1555, 0.1799] | 0.9002 | Not met |
| STID | PEMS-BAY | C4 | 0.0673 [-0.1183, 0.2755] | 0.0754 [-0.1217, 0.2974] | 0.9062 | Not met |
| DCRNN | METR-LA | C3 | 0.9825 [0.4790, 1.5777] | 0.9924 [0.4901, 1.5782] | 0.8891 | Met |
| DCRNN | METR-LA | C4 | 1.4414 [0.8521, 2.1265] | 1.4697 [0.8682, 2.1660] | 0.8943 | Met |
| DCRNN | PEMS-BAY | C3 | 0.5709 [0.3503, 0.8975] | 0.5343 [0.3292, 0.8480] | 0.9007 | Met |
| DCRNN | PEMS-BAY | C4 | 0.7216 [0.4733, 0.9883] | 0.7034 [0.4623, 0.9590] | 0.9144 | Met |

The registered superiority rule was met in **12 of 16** backbone/dataset/condition cells: 2/4 with FAR-GW, 4/4 with STAEformer, 2/4 with STID, and 4/4 with DCRNN. All four tested backbones met the rule on METR-LA C3 and C4. STAEformer and DCRNN also met it on PEMS-BAY C3 and C4, whereas FAR-GW and STID did not.

This is evidence that the method can transfer across multiple frozen forecasting backbones on METR-LA. It does not support a universal backbone-independent superiority claim: the PEMS-BAY result changes with the frozen backbone, and the audits do not include a formal interaction test between backbone and calibrator effects.

The direct CoRel adaptation remains a separate calibration-method comparison on the FAR-GW prediction caches; it is not evidence from an additional point backbone.

Machine-readable evidence: `artifacts/evaluation/cross-backbone-transfer-synthesis-v3.json`.
