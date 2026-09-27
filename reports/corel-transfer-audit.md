# CoRel feedback-arrival transfer audit

All 36 registered cells completed: two datasets, C3/C4, three model seeds, and
three fault seeds. CoRel used the official architecture pinned at commit
`4504c4edf76128bfe657762a41ec9eb043038ca2`, identical frozen FAR-GW point
forecasts, hourly fault-focused origins, and the same evaluation masks as FAR-Cal,
rolling, and ACI. Feedback released at bin `t` enters the CoRel residual window
for forecasts from `t+1`, matching its training-window convention.

| Dataset / condition | FAR-Cal score | CoRel score | CoRel − FAR-Cal (95% CI) | FAR-Cal coverage | CoRel coverage |
|---|---:|---:|---:|---:|---:|
| METR-LA C3 | 20.3370 | 22.0330 | 1.6960 [1.1594, 2.3049] | 0.8985 | 0.8743 |
| METR-LA C4 | 23.4332 | 25.6099 | 2.1767 [1.3784, 3.1390] | 0.8958 | 0.8725 |
| PEMS-BAY C3 | 10.3271 | 12.5817 | 2.2547 [1.8212, 2.7181] | 0.9006 | 0.8837 |
| PEMS-BAY C4 | 10.5644 | 12.8094 | 2.2450 [1.7615, 2.7452] | 0.9068 | 0.8743 |

Positive differences favor FAR-Cal. Every lower confidence bound is above zero,
so FAR-Cal has lower mean 90% interval score than the CoRel adaptation in all four
audits. FAR-Cal also clears the 0.88 coverage floor in all four. CoRel clears it
only for PEMS-BAY C3.

CoRel was worse than rolling and ACI in all four audits. This does not overturn
the earlier PEMS-BAY finding: rolling and ACI still outperform FAR-Cal there.
The result establishes superiority over this direct relational comparator within
the registered PS-1 multihorizon feedback adaptation. It is not a claim about
CoRel's published source split or universal method quality.

Inference uses paired one-day moving blocks over hourly origins, stratified by
fault seed after averaging the three model seeds. All 36 manifests and chunk
hashes were verified; the repository has 71 passing tests.
