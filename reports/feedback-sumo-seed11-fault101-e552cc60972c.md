# SUMO paired feedback timing — model seed 11, fault seed 101

Evaluation SHA-256: `e552cc60972cef2fdb983ff0595299588aea24a231fcc022f45bcbe5085ddac6`

Point/input scenario: C4. Feedback comparison: C4I (immediate) versus C4 (delayed).

Positive differences mean delayed feedback has a worse interval score.

| Method | Equal-horizon 90% score difference | 95% episode-bootstrap interval |
|---|---:|---:|
| aci | 0.0012064 | [-0.0003551, 0.0032689] |
| rolling | 0.0004802 | [-0.0001678, 0.0015590] |

This is preliminary H1 evidence only. One model seed; One fault seed; Rolling and project-variant ACI only; H1 evidence incomplete until registered replication.
