# SUMO paired feedback timing — model seed 33, fault seed 202

Evaluation SHA-256: `cb66f7e057ff664c4362cdf2da1ade9aa14e6442bafbacefed3f825bd5fafbb7`

Point/input scenario: C4. Feedback comparison: C4I (immediate) versus C4 (delayed).

Positive differences mean delayed feedback has a worse interval score.

| Method | Equal-horizon 90% score difference | 95% episode-bootstrap interval |
|---|---:|---:|
| aci | 0.0016190 | [-0.0002209, 0.0046868] |
| rolling | 0.0011745 | [-0.0002248, 0.0035975] |

This is preliminary H1 evidence only. One model seed; One fault seed; Rolling and project-variant ACI only; H1 evidence incomplete until registered replication.
