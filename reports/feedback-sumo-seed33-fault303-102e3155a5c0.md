# SUMO paired feedback timing — model seed 33, fault seed 303

Evaluation SHA-256: `102e3155a5c0d408b298df59497220e56e9529e65dcbdb743a26a27f0f6b668e`

Point/input scenario: C3. Feedback comparison: C2 (immediate) versus C3 (delayed).

Positive differences mean delayed feedback has a worse interval score.

| Method | Equal-horizon 90% score difference | 95% episode-bootstrap interval |
|---|---:|---:|
| aci | -0.0007292 | [-0.0022980, 0.0005959] |
| rolling | 0.0003308 | [-0.0003799, 0.0014027] |

This is preliminary H1 evidence only. One model seed; One fault seed; Rolling and project-variant ACI only; H1 evidence incomplete until registered replication.
