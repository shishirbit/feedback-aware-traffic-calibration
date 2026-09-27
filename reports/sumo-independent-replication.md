# Independent SUMO replication

Regularized FAR-Cal candidate `v2s25` was locked before generating 30 new SUMO episodes (seeds 50001–50030). Results average paired effects across three model seeds and three C3 fault seeds within each episode, followed by a 2,000-resample episode bootstrap. Positive differences favor FAR-Cal.

| Result | Estimate | 95% episode-bootstrap interval |
|---|---:|---:|
| Rolling − FAR-Cal interval score | 0.0517 | [-0.0480, 0.1477] |
| ACI − FAR-Cal interval score | 0.0376 | [-0.0502, 0.1157] |

FAR-Cal's aggregate equal-horizon 90% interval score was 10.1265 and its coverage was 0.8907, above the registered 0.88 floor. It achieved lower descriptive interval score than both comparators, but both confidence intervals include zero. The registered superiority rule was therefore not met.

Machine-readable results: `artifacts/evaluation/sumo-independent-replication.json`.
