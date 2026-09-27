# SUMO paired feedback timing — model seed 22, fault seed 202

Evaluation SHA-256: `0b8c5895eef6471782a826970b9a0c0f97d058c69de87c772faa1cacd24dae47`

Point/input scenario: C4. Feedback comparison: C4I (immediate) versus C4 (delayed).

Positive differences mean delayed feedback has a worse interval score.

| Method | Equal-horizon 90% score difference | 95% episode-bootstrap interval |
|---|---:|---:|
| aci | 0.0006268 | [-0.0003360, 0.0018700] |
| rolling | 0.0002532 | [-0.0004476, 0.0013078] |

This is preliminary H1 evidence only. One model seed; One fault seed; Rolling and project-variant ACI only; H1 evidence incomplete until registered replication.
