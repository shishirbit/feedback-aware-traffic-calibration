# PEMS_BAY paired feedback timing — model seed 22, fault seed 101

Evaluation SHA-256: `4f3936ca86ecf06800963f843aec6f29e710669ef8bd8ab3e3be5c7a9a1d0ff8`

Point/input scenario: C3. Feedback comparison: C2 (immediate) versus C3 (delayed).

Positive differences mean delayed feedback has a worse interval score.

| Method | Equal-horizon 90% score difference | 95% block-bootstrap interval |
|---|---:|---:|
| aci | 0.0000911 | [0.0000344, 0.0001542] |
| rolling | 0.0000204 | [0.0000049, 0.0000355] |

This is preliminary H1 evidence only. One model seed; One fault seed; Rolling and project-variant ACI only; H1 evidence incomplete until registered replication.
