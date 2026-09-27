# PS-1 evidence status

Registry SHA-256: `d12d58b9a7fa736828e4b51f1535e454768fbd0db2c6b1721d48e364b541378c`

This report distinguishes executed software checks from pending research evidence.

| Experiment | Dataset | Status | Evidence |
|---|---|---|---|
| smoke_cpu | sumo_synthetic_motorway | completed | artifacts/smoke-66538f1f9de7 |
| metr_la_core | metr_la | planned | resumable predictor training is ready but full CPU run and replay are not yet executed |
| pems_bay_core | pems_bay | planned | resumable predictor training is ready but full CPU run and replay are not yet executed |
| metr_la_cpu_feasibility | metr_la | completed | artifacts/feasibility/metr_la-seed11.json |
| metr_la_far_gw_seed11_training | metr_la | completed | artifacts/training/metr_la-seed11-803c69399a64/training_summary.json |
| metr_la_far_gw_seed22_training | metr_la | running | artifacts/training/metr_la-seed22-803c69399a64/run_manifest.json |
| metr_la_seed11_clean_c0 | metr_la | completed | artifacts/evaluation/metr_la-seed11-clean.json |
| pems_bay_cpu_feasibility | pems_bay | completed | artifacts/feasibility/pems_bay-seed11.json |
| sumo_core | sumo | planned | P1-P3 generation, three-seed training, fault replay, and episode bootstrap not yet executed |
| sumo_live_smoke | sumo_synthetic_motorway | completed | data/sumo/live-smoke/live_equivalence.json |

## Scientific claim status

H1-H4: **not evaluated**. Completed clean, smoke, training, and integration entries do not test the registered fault hypotheses.

No missing result has been replaced by a generated number or zero.
