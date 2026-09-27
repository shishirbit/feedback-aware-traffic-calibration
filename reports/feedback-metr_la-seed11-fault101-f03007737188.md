# METR_LA paired feedback timing — model seed 11, fault seed 101

Evaluation SHA-256: `f030077371882b5f77ff2984875d37d093ef9abb71b25a96028edfa9ea2d9c29`

Point/input scenario: C3. Feedback comparison: C2 (immediate) versus C3 (delayed).

Positive differences mean delayed feedback has a worse interval score.

| Method | Equal-horizon 90% score difference | 95% block-bootstrap interval |
|---|---:|---:|
| aci | -0.0001035 | [-0.0003691, 0.0000833] |
| rolling | 0.0000395 | [0.0000074, 0.0000756] |

This is preliminary H1 evidence only. One model seed; One fault seed; Rolling and project-variant ACI only; H1 evidence incomplete until registered replication.
