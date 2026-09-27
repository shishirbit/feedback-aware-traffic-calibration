# PS-1 evidence status

Registry SHA-256: `a04049c55df12ef3894be43e5b1c536db0a439e4bab649af646799eaa6075703`

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
| metr_la_simple_controls_clean_c0 | metr_la | completed | artifacts/evaluation/metr_la-simple-controls-clean.json |
| pems_bay_cpu_feasibility | pems_bay | completed | artifacts/feasibility/pems_bay-seed11.json |
| pems_bay_simple_controls_clean_c0 | pems_bay | completed | artifacts/evaluation/pems_bay-simple-controls-clean.json |
| sumo_core | sumo | blocked | Installed SUMO executables stopped launching after restart with Windows integrity status 0xc0e90002; P1-P3 generation cannot proceed until a runnable build is available |
| sumo_live_smoke | sumo_synthetic_motorway | completed | data/sumo/live-smoke/live_equivalence.json |

## Scientific claim status

H1-H4: **not evaluated**. Completed clean, smoke, training, and integration entries do not test the registered fault hypotheses.

No missing result has been replaced by a generated number or zero.
