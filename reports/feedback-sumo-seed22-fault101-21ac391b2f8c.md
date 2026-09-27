# SUMO paired feedback timing — model seed 22, fault seed 101

Evaluation SHA-256: `21ac391b2f8cc86bb4a9cad1a42311f92e5a5049c7180d2a4c17f6b78847df5a`

Point/input scenario: C4. Feedback comparison: C4I (immediate) versus C4 (delayed).

Positive differences mean delayed feedback has a worse interval score.

| Method | Equal-horizon 90% score difference | 95% episode-bootstrap interval |
|---|---:|---:|
| aci | -0.0006155 | [-0.0029742, 0.0011019] |
| rolling | -0.0000088 | [-0.0004456, 0.0004248] |

This is preliminary H1 evidence only. One model seed; One fault seed; Rolling and project-variant ACI only; H1 evidence incomplete until registered replication.
