# SUMO paired feedback timing — model seed 11, fault seed 101

Evaluation SHA-256: `c6f9cf3f1ce5603257c17e3ffaa3646a2fd184926d1a6e0d24193f925a14d139`

Point/input scenario: C3. Feedback comparison: C2 (immediate) versus C3 (delayed).

Positive differences mean delayed feedback has a worse interval score.

| Method | Equal-horizon 90% score difference | 95% episode-bootstrap interval |
|---|---:|---:|
| aci | 0.0011393 | [-0.0004644, 0.0032461] |
| rolling | 0.0004833 | [-0.0000396, 0.0014655] |

This is preliminary H1 evidence only. One model seed; One fault seed; Rolling and project-variant ACI only; H1 evidence incomplete until registered replication.
