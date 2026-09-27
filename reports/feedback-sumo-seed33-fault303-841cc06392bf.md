# SUMO paired feedback timing — model seed 33, fault seed 303

Evaluation SHA-256: `841cc06392bff44d0c42761c442f55e9a44b87b3ee94acfeffb4aa4aa0dccd93`

Point/input scenario: C4. Feedback comparison: C4I (immediate) versus C4 (delayed).

Positive differences mean delayed feedback has a worse interval score.

| Method | Equal-horizon 90% score difference | 95% episode-bootstrap interval |
|---|---:|---:|
| aci | -0.0009884 | [-0.0029646, 0.0005513] |
| rolling | 0.0006830 | [-0.0004457, 0.0022418] |

This is preliminary H1 evidence only. One model seed; One fault seed; Rolling and project-variant ACI only; H1 evidence incomplete until registered replication.
