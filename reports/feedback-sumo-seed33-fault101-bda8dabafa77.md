# SUMO paired feedback timing — model seed 33, fault seed 101

Evaluation SHA-256: `bda8dabafa77b9a32ebfb423da74c366c4eedc8f767fdf1f43c5b56c8d110786`

Point/input scenario: C4. Feedback comparison: C4I (immediate) versus C4 (delayed).

Positive differences mean delayed feedback has a worse interval score.

| Method | Equal-horizon 90% score difference | 95% episode-bootstrap interval |
|---|---:|---:|
| aci | 0.0023573 | [0.0000500, 0.0065004] |
| rolling | 0.0000613 | [-0.0007160, 0.0009214] |

This is preliminary H1 evidence only. One model seed; One fault seed; Rolling and project-variant ACI only; H1 evidence incomplete until registered replication.
