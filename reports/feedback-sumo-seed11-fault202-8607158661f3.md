# SUMO paired feedback timing — model seed 11, fault seed 202

Evaluation SHA-256: `8607158661f3fdc181b72d23f6196feaf087410a46797a259fdc6a4d222f4f62`

Point/input scenario: C3. Feedback comparison: C2 (immediate) versus C3 (delayed).

Positive differences mean delayed feedback has a worse interval score.

| Method | Equal-horizon 90% score difference | 95% episode-bootstrap interval |
|---|---:|---:|
| aci | 0.0000984 | [-0.0026714, 0.0023003] |
| rolling | -0.0004239 | [-0.0018138, 0.0004560] |

This is preliminary H1 evidence only. One model seed; One fault seed; Rolling and project-variant ACI only; H1 evidence incomplete until registered replication.
