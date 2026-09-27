# SUMO paired feedback timing — model seed 11, fault seed 303

Evaluation SHA-256: `4bb4e4b7d461e1ddff035a1f5c31399e48da971b0dae86515d8092247622e8c2`

Point/input scenario: C3. Feedback comparison: C2 (immediate) versus C3 (delayed).

Positive differences mean delayed feedback has a worse interval score.

| Method | Equal-horizon 90% score difference | 95% episode-bootstrap interval |
|---|---:|---:|
| aci | -0.0027387 | [-0.0065042, -0.0001357] |
| rolling | -0.0002155 | [-0.0018573, 0.0010707] |

This is preliminary H1 evidence only. One model seed; One fault seed; Rolling and project-variant ACI only; H1 evidence incomplete until registered replication.
