# Locked FAR-Cal held-out evaluation

Candidate `n2b005w2` was locked before this evaluation. A later protocol audit found that the original tuning implementation had incorrectly used the future calibration archive. Correct split-safe tuning used tune episodes 60–64 as temporary residual warm-up and scored episodes 65–74; it independently selected the same candidate, without changing any test parameter. This post-test audit does not erase the protocol deviation. The held-out C3 test covers 30 episodes for each of three model seeds and three fault seeds. Positive differences mean the comparator has a worse equal-horizon 90% interval score than FAR-Cal.

| Model seed | Fault seed | FAR-Cal | Rolling − FAR-Cal (95% episode CI) | ACI − FAR-Cal (95% episode CI) |
|---:|---:|---:|---:|---:|
| 11 | 101 | 12.33 | -0.13 [-0.32, 0.03] | -0.21 [-0.44, -0.01] |
| 11 | 202 | 12.32 | -0.12 [-0.30, 0.04] | -0.20 [-0.43, -0.00] |
| 11 | 303 | 12.27 | -0.11 [-0.29, 0.04] | -0.20 [-0.43, -0.01] |
| 22 | 101 | 12.76 | -0.33 [-0.90, 0.07] | -0.36 [-0.80, -0.00] |
| 22 | 202 | 12.76 | -0.33 [-0.90, 0.08] | -0.36 [-0.81, -0.00] |
| 22 | 303 | 12.75 | -0.33 [-0.90, 0.07] | -0.36 [-0.81, -0.00] |
| 33 | 101 | 12.38 | -0.01 [-0.17, 0.11] | -0.15 [-0.35, 0.04] |
| 33 | 202 | 12.37 | -0.00 [-0.15, 0.12] | -0.14 [-0.33, 0.05] |
| 33 | 303 | 12.33 | 0.00 [-0.13, 0.11] | -0.14 [-0.33, 0.04] |

The mean effect across the nine runs is -0.150 for rolling and -0.234 for ACI. No run has a 95% interval wholly above zero. The locked FAR-Cal configuration therefore does not establish superiority over either comparator on this held-out test.

Machine-readable results: `artifacts/evaluation/sumo-C3-farcal-locked-full.json`.
