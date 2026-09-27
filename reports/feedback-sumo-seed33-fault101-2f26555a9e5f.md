# SUMO paired feedback timing — model seed 33, fault seed 101

Evaluation SHA-256: `2f26555a9e5fa5d45bfad3ffdc7982be3a6d1868d0d8e9a5d0d4f7745788c0e5`

Point/input scenario: C3. Feedback comparison: C2 (immediate) versus C3 (delayed).

Positive differences mean delayed feedback has a worse interval score.

| Method | Equal-horizon 90% score difference | 95% episode-bootstrap interval |
|---|---:|---:|
| aci | 0.0024625 | [0.0002192, 0.0063487] |
| rolling | 0.0002794 | [-0.0000934, 0.0009668] |

This is preliminary H1 evidence only. One model seed; One fault seed; Rolling and project-variant ACI only; H1 evidence incomplete until registered replication.
