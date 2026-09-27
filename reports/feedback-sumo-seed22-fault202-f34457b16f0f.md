# SUMO paired feedback timing — model seed 22, fault seed 202

Evaluation SHA-256: `f34457b16f0fbca65a44fb221816c07ba08a1bb3f124686ea881d14c79a45114`

Point/input scenario: C3. Feedback comparison: C2 (immediate) versus C3 (delayed).

Positive differences mean delayed feedback has a worse interval score.

| Method | Equal-horizon 90% score difference | 95% episode-bootstrap interval |
|---|---:|---:|
| aci | 0.0013566 | [0.0002278, 0.0027201] |
| rolling | 0.0006421 | [-0.0000134, 0.0017904] |

This is preliminary H1 evidence only. One model seed; One fault seed; Rolling and project-variant ACI only; H1 evidence incomplete until registered replication.
