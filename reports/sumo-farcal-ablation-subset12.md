# SUMO FAR-Cal ablation — prespecified 12-episode subset

Evaluation SHA-256: `2c57e3ee115a3925d292b9fb5da4dfb38d23af60c36f1c90e4eefa5d58acb51e`

C3 fault seed 101 was evaluated on test-001 through test-012, with three episodes from each P0–P3 family. Positive differences mean the comparator has a worse equal-horizon 90% interval score than FAR-Cal.

| Model seed | FAR-Cal score | Rolling minus FAR-Cal | 95% episode interval | ACI minus FAR-Cal | 95% episode interval |
|---:|---:|---:|---:|---:|---:|
| 11 | 13.9874 | -0.2924 | [-0.6753, 0.0120] | -0.3598 | [-0.7981, 0.0115] |
| 22 | 14.6368 | -0.6916 | [-1.9347, 0.1173] | -0.5720 | [-1.4441, 0.0724] |
| 33 | 13.6305 | -0.1201 | [-0.3858, 0.0809] | -0.3234 | [-0.7178, 0.0364] |

Rolling and ACI are descriptively better on this subset, although all three per-seed intervals cross zero. Removing age weighting improves the descriptive score most among the five ablations. These are preliminary subset results and do not confirm a FAR-Cal advantage.
