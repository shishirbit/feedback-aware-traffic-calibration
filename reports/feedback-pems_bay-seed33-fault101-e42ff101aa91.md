# PEMS_BAY paired feedback timing — model seed 33, fault seed 101

Evaluation SHA-256: `e42ff101aa91508a176de1425d34d75c72ead2675913553da993e60aaddb76ac`

Point/input scenario: C3. Feedback comparison: C2 (immediate) versus C3 (delayed).

Positive differences mean delayed feedback has a worse interval score.

| Method | Equal-horizon 90% score difference | 95% block-bootstrap interval |
|---|---:|---:|
| aci | 0.0000298 | [-0.0000107, 0.0000597] |
| rolling | 0.0000203 | [0.0000074, 0.0000324] |

This is preliminary H1 evidence only. One model seed; One fault seed; Rolling and project-variant ACI only; H1 evidence incomplete until registered replication.
