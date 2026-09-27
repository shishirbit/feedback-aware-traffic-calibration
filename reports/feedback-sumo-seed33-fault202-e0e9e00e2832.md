# SUMO paired feedback timing — model seed 33, fault seed 202

Evaluation SHA-256: `e0e9e00e2832c7a019a3760c93c5bfa48db844ed2bf509cfbfa0d1b36b420db0`

Point/input scenario: C3. Feedback comparison: C2 (immediate) versus C3 (delayed).

Positive differences mean delayed feedback has a worse interval score.

| Method | Equal-horizon 90% score difference | 95% episode-bootstrap interval |
|---|---:|---:|
| aci | 0.0018637 | [-0.0002561, 0.0054259] |
| rolling | 0.0002973 | [-0.0000742, 0.0009699] |

This is preliminary H1 evidence only. One model seed; One fault seed; Rolling and project-variant ACI only; H1 evidence incomplete until registered replication.
