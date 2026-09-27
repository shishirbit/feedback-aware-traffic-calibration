# Powered SUMO confirmation

Regularized FAR-Cal candidate `v2s25` was fixed before generating 160 fresh SUMO episodes (seeds 60001–60160), balanced equally across P0–P3. Paired effects were averaged across three model seeds and three C3 fault seeds within each episode, followed by a 2,000-resample episode bootstrap. Positive differences favor FAR-Cal.

| Result | Estimate | 95% episode-bootstrap interval |
|---|---:|---:|
| Rolling − FAR-Cal interval score | -0.0270 | [-0.1057, 0.0368] |
| ACI − FAR-Cal interval score | -0.0262 | [-0.1039, 0.0391] |

FAR-Cal's aggregate equal-horizon 90% interval score was 10.0577 and its coverage was 0.8953, above the registered 0.88 floor. The small point estimates favor the comparators, and both confidence intervals include zero. The registered superiority rule was not met. Together with the earlier 30-episode replication, this does not support a FAR-Cal superiority claim.

Machine-readable results: `artifacts/evaluation/sumo-powered-confirmation.json`.
