# Full registered component ablations

All 432 stages completed on unchanged FAR-GW prediction caches.

Positive paired interval-score differences mean removing the component worsened performance. Intervals are exploratory 95% bootstrap intervals; overlapping or negative intervals are reported without a superiority claim.

Real-data inference uses day blocks within fault seeds after averaging model seeds. SUMO inference resamples episodes after averaging model and fault seeds. C3 uses all 40 fresh validation episodes; C4 uses all 30 test episodes.

| Dataset | Scenario | Ablation | Score difference (95% CI) | Coverage | Width (mph) |
|---|---|---|---|---|---|
| metr_la | C3 | No spatial pooling | 0.349 [0.147, 0.611] | 0.890 | 11.061 |
| metr_la | C3 | No context matching | 0.001 [-0.088, 0.090] | 0.900 | 11.002 |
| metr_la | C3 | No target-age weights | -0.040 [-0.252, 0.167] | 0.903 | 10.906 |
| metr_la | C3 | No stale blend | 0.297 [0.145, 0.502] | 0.899 | 11.191 |
| metr_la | C3 | No inflation | 0.010 [-0.005, 0.031] | 0.898 | 10.888 |
| metr_la | C3 | Arrival-time freshness | 0.003 [-0.004, 0.009] | 0.899 | 10.929 |
| metr_la | C3 | Frozen archive only | 0.305 [-0.087, 0.734] | 0.883 | 10.021 |
| metr_la | C4 | No spatial pooling | 0.223 [-0.138, 0.600] | 0.889 | 11.724 |
| metr_la | C4 | No context matching | -0.062 [-0.150, 0.025] | 0.896 | 11.600 |
| metr_la | C4 | No target-age weights | -0.252 [-0.462, -0.072] | 0.900 | 11.503 |
| metr_la | C4 | No stale blend | 0.106 [-0.081, 0.283] | 0.896 | 11.884 |
| metr_la | C4 | No inflation | 0.040 [0.012, 0.068] | 0.894 | 11.419 |
| metr_la | C4 | Arrival-time freshness | 0.008 [0.004, 0.014] | 0.896 | 11.557 |
| metr_la | C4 | Frozen archive only | 0.148 [-0.352, 0.672] | 0.880 | 10.476 |
| pems_bay | C3 | No spatial pooling | 0.154 [0.065, 0.269] | 0.894 | 6.319 |
| pems_bay | C3 | No context matching | 0.000 [-0.002, 0.002] | 0.901 | 6.261 |
| pems_bay | C3 | No target-age weights | -0.120 [-0.189, -0.025] | 0.905 | 6.253 |
| pems_bay | C3 | No stale blend | 0.018 [0.005, 0.027] | 0.899 | 6.255 |
| pems_bay | C3 | No inflation | 0.000 [-0.000, 0.001] | 0.900 | 6.256 |
| pems_bay | C3 | Arrival-time freshness | 0.003 [0.001, 0.006] | 0.900 | 6.255 |
| pems_bay | C3 | Frozen archive only | 0.208 [0.094, 0.360] | 0.908 | 6.421 |
| pems_bay | C4 | No spatial pooling | 0.276 [0.156, 0.389] | 0.900 | 7.000 |
| pems_bay | C4 | No context matching | -0.226 [-0.395, -0.084] | 0.902 | 6.662 |
| pems_bay | C4 | No target-age weights | -0.028 [-0.064, -0.001] | 0.907 | 6.780 |
| pems_bay | C4 | No stale blend | 9.441 [8.410, 10.511] | 0.549 | 3.750 |
| pems_bay | C4 | No inflation | -0.000 [-0.007, 0.006] | 0.904 | 6.709 |
| pems_bay | C4 | Arrival-time freshness | 0.000 [-0.001, 0.001] | 0.907 | 6.775 |
| pems_bay | C4 | Frozen archive only | 0.039 [-0.009, 0.077] | 0.905 | 6.771 |
| sumo | C3 | No spatial pooling | 0.015 [-0.025, 0.053] | 0.888 | 4.658 |
| sumo | C3 | No context matching | 0.003 [-0.003, 0.011] | 0.889 | 4.652 |
| sumo | C3 | No target-age weights | 0.097 [-0.069, 0.248] | 0.899 | 5.124 |
| sumo | C3 | No stale blend | 0.075 [0.049, 0.103] | 0.882 | 4.569 |
| sumo | C3 | No inflation | 0.001 [-0.005, 0.007] | 0.888 | 4.632 |
| sumo | C3 | Arrival-time freshness | 0.000 [-0.000, 0.000] | 0.889 | 4.656 |
| sumo | C3 | Frozen archive only | 0.109 [-0.116, 0.314] | 0.904 | 5.283 |
| sumo | C4 | No spatial pooling | 0.011 [-0.030, 0.052] | 0.878 | 5.385 |
| sumo | C4 | No context matching | 0.043 [0.008, 0.087] | 0.879 | 5.419 |
| sumo | C4 | No target-age weights | -0.246 [-0.637, 0.074] | 0.892 | 5.724 |
| sumo | C4 | No stale blend | 0.499 [0.245, 0.862] | 0.858 | 5.216 |
| sumo | C4 | No inflation | 0.001 [-0.006, 0.009] | 0.878 | 5.380 |
| sumo | C4 | Arrival-time freshness | 0.000 [-0.000, 0.000] | 0.879 | 5.405 |
| sumo | C4 | Frozen archive only | -0.233 [-0.635, 0.098] | 0.896 | 5.855 |

The arrival-time diagnostic changes freshness weighting and the stale gap only; buffer eligibility and ESS blocks remain target-based. The frozen-only comparator removes live adaptation and adaptive inflation.

These results concern FAR-GW component ablations. The separately completed four-backbone main comparison includes STAEformer. No universal superiority or identifiable MNAR correction is claimed.

Completion audit verified all 432 manifest hashes and 5,328 chunk hashes. Across 42 exploratory intervals, 13 removals worsen interval score, four improve it and 25 include zero. These are unadjusted exploratory comparisons, not confirmatory multiple-testing results.
