# SUMO paired feedback timing — model seed 22, fault seed 101

Evaluation SHA-256: `a1a682d26662b93fbfdbed36dc963df3a28b9a662f2d18fea96b871b1ccdb1af`

Point/input scenario: C3. Feedback comparison: C2 (immediate) versus C3 (delayed).

Positive differences mean delayed feedback has a worse interval score.

| Method | Equal-horizon 90% score difference | 95% episode-bootstrap interval |
|---|---:|---:|
| aci | 0.0010496 | [-0.0004911, 0.0028477] |
| rolling | 0.0001179 | [-0.0000583, 0.0004242] |

This is preliminary H1 evidence only. One model seed; One fault seed; Rolling and project-variant ACI only; H1 evidence incomplete until registered replication.
