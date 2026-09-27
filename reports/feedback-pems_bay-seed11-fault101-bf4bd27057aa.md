# PEMS_BAY paired feedback timing — model seed 11, fault seed 101

Evaluation SHA-256: `bf4bd27057aa1f0ec5ef93f0e34d7309db207ea217bde4bd861366b2ef10d8ab`

Point/input scenario: C3. Feedback comparison: C2 (immediate) versus C3 (delayed).

Positive differences mean delayed feedback has a worse interval score.

| Method | Equal-horizon 90% score difference | 95% block-bootstrap interval |
|---|---:|---:|
| aci | 0.0000425 | [-0.0000083, 0.0000970] |
| rolling | 0.0000205 | [0.0000087, 0.0000319] |

This is preliminary H1 evidence only. One model seed; One fault seed; Rolling and project-variant ACI only; H1 evidence incomplete until registered replication.
