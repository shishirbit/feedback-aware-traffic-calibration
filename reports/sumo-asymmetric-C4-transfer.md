# SUMO asymmetric FAR-Cal C4 transfer

The C3-locked `asym100` candidate was transferred to the existing held-out SUMO C4 permanent-feedback-loss condition across three model seeds, three fault seeds, and 30 episodes. Positive paired differences favor asymmetric FAR-Cal.

| Result | Estimate | 95% episode-bootstrap interval |
|---|---:|---:|
| Rolling − asymmetric FAR-Cal interval score | 1.0285 | [0.6083, 1.5249] |
| ACI − asymmetric FAR-Cal interval score | 0.9435 | [0.5728, 1.3726] |

The method's aggregate 90% interval score was 11.2948 and coverage was 0.8793. Although both score intervals exclude zero in its favor, coverage missed the registered 0.88 floor by 0.0007. The complete superiority rule was therefore not met.

Machine-readable results: `artifacts/evaluation/sumo-asymmetric-C4-transfer.json`.
