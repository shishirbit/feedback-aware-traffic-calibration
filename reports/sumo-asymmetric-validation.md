# Asymmetric FAR-Cal validation

The asymmetric FAR-Cal candidate `asym100` was fixed before generating 40 fresh SUMO episodes (seeds 70001–70040), balanced across P0–P3. The registered analysis averaged paired effects across three model seeds and three C3 fault seeds within each episode, then used a 2,000-resample episode bootstrap. Positive differences favor asymmetric FAR-Cal.

| Result | Estimate | 95% episode-bootstrap interval |
|---|---:|---:|
| Rolling − asymmetric FAR-Cal interval score | 0.5947 | [0.2356, 1.0065] |
| ACI − asymmetric FAR-Cal interval score | 0.5788 | [0.2319, 0.9544] |

Asymmetric FAR-Cal's aggregate equal-horizon 90% interval score was 7.6038 and coverage was 0.8890, above the registered 0.88 floor. Both confidence intervals exclude zero in favor of asymmetric FAR-Cal, so the registered superiority rule was met on this fresh SUMO C3 validation.

Machine-readable results: `artifacts/evaluation/sumo-asymmetric-validation.json`.
