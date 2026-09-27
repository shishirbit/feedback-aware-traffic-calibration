# PS-1 evidence status

Registry SHA-256: `09a1edcbbe0e506bd86972d8692d069cb83ef9664c5dd7e8e7be903d7a09531b`

This report distinguishes executed software checks from pending research evidence.

| Experiment | Dataset | Status | Evidence |
|---|---|---|---|
| smoke_cpu | sumo_synthetic_motorway | completed | artifacts/smoke-66538f1f9de7 |
| metr_la_core | metr_la | planned | all three registered predictor seeds are trained; full scenario replay and calibration comparison remain |
| pems_bay_core | pems_bay | running | first registered predictor seed is training on CPU; replay follows its verified checkpoint |
| metr_la_cpu_feasibility | metr_la | completed | artifacts/feasibility/metr_la-seed11.json |
| metr_la_far_gw_seed11_training | metr_la | completed | artifacts/training/metr_la-seed11-803c69399a64/training_summary.json |
| metr_la_far_gw_seed22_training | metr_la | completed | artifacts/training/metr_la-seed22-803c69399a64/training_summary.json |
| metr_la_far_gw_seed33_training | metr_la | completed | artifacts/training/metr_la-seed33-803c69399a64/training_summary.json |
| metr_la_seed11_clean_c0 | metr_la | completed | artifacts/evaluation/metr_la-seed11-clean.json |
| metr_la_seed22_clean_c0 | metr_la | completed | artifacts/evaluation/metr_la-seed22-clean.json |
| metr_la_seed11_c0_replay_validation | metr_la | completed | artifacts/validation/metr_la-seed11-C0-fault101.json |
| metr_la_seed11_c3_fault101_static | metr_la | completed | artifacts/evaluation/metr_la-seed11-C3-fault101-static.json |
| metr_la_seed11_c3_inputs_c2_feedback_fault101_online | metr_la | completed | artifacts/online/metr_la-seed11-C3-feedback-C2-fault101/manifest.json |
| metr_la_seed11_c3_inputs_c3_feedback_fault101_online | metr_la | completed | artifacts/online/metr_la-seed11-C3-feedback-C3-fault101/manifest.json |
| metr_la_seed11_feedback_timing_pair_fault101 | metr_la | completed | artifacts/evaluation/metr_la-seed11-C3-feedback-C2-vs-C3-fault101.json |
| metr_la_seed22_c3_fault101_static | metr_la | completed | artifacts/evaluation/metr_la-seed22-C3-fault101-static.json |
| metr_la_seed22_c3_inputs_c2_feedback_fault101_online | metr_la | completed | artifacts/online/metr_la-seed22-C3-feedback-C2-fault101/manifest.json |
| metr_la_seed22_c3_inputs_c3_feedback_fault101_online | metr_la | completed | artifacts/online/metr_la-seed22-C3-feedback-C3-fault101/manifest.json |
| metr_la_seed22_feedback_timing_pair_fault101 | metr_la | completed | artifacts/evaluation/metr_la-seed22-C3-feedback-C2-vs-C3-fault101.json |
| metr_la_simple_controls_clean_c0 | metr_la | completed | artifacts/evaluation/metr_la-simple-controls-clean.json |
| pems_bay_cpu_feasibility | pems_bay | completed | artifacts/feasibility/pems_bay-seed11.json |
| pems_bay_far_gw_seed11_training | pems_bay | completed | artifacts/training/pems_bay-seed11-310e9ba8c418/training_summary.json |
| pems_bay_far_gw_seed22_training | pems_bay | running | artifacts/training/pems_bay-seed22-310e9ba8c418/run_manifest.json |
| pems_bay_simple_controls_clean_c0 | pems_bay | completed | artifacts/evaluation/pems_bay-simple-controls-clean.json |
| sumo_core | sumo | blocked | Installed SUMO executables stopped launching after restart with Windows integrity status 0xc0e90002; P1-P3 generation cannot proceed until a runnable build is available |
| sumo_live_smoke | sumo_synthetic_motorway | completed | data/sumo/live-smoke/live_equivalence.json |

## Scientific claim status

H1 has preliminary artifact-backed evidence where listed; it is not confirmed until all registered model/fault-seed replication and methods are complete. H2-H4 are not evaluated.

No missing result has been replaced by a generated number or zero.
