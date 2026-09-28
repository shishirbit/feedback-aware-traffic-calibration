# Feedback-aware calibration of traffic prediction intervals under delayed and missing sensor reports

Reproducibility release

This release accompanies the research manuscript by Shishir Singh Chauhan, Manipal University Jaipur. It contains code, configurations, tests, generated SUMO prepared arrays, evidence summaries and the supplied-template manuscript.

The registered evaluations and all 432 ablation stages are complete. Findings are conditional: no universal superiority, no identifiable MNAR correction and no production-service claim. Negative and inconclusive results are retained.

## Reproduce

Install the relevant locked environment (Python 3.11 backbone/CoRel locks or the main lock), install this package with `pip install -e .`, and run `python -m unittest discover -s tests`. The latest local verification passed 77 tests. SUMO/TraCI 1.27.1 is required for simulation regeneration. Raw real benchmark packages must be acquired separately using `docs/sources.md` and checked against the recorded hashes; graphs are obtained from pinned DCRNN sources. Training checkpoints and bulky prediction/residual caches are not bundled: regenerate these before invoking resumable evaluation runners. The original protocols, candidate locks, inference units and limitations are in docs and reports.

Generated SUMO prepared arrays cover original, replication, confirmation and fresh asymmetric-validation sets; original episode XML can be regenerated from registered plans. Prepared manifest provenance paths describe the original host and are not portable runtime paths. This is an evidence/source release rather than a precomputed-cache image. Study scripts use repository-relative paths unless explicitly documented as original-host launch helpers.

The authoritative manuscript is a single `manuscript/main.tex`, containing text, bibliography, tables and captions, with external figure PDFs. Compile it directly with Tectonic or LaTeX; no BibTeX step is required. Older migration/asset scripts are not a manuscript reconstruction command. All manuscript effects are converted from native mph artifacts to SI using 0.44704 exactly.

## Original workflow and command inventory

# PS-1 feedback-aware traffic uncertainty

The publication-ready figure registry and generated IEEE-sized outputs are documented in [reports/publication-figure-suite.md](reports/publication-figure-suite.md). Rebuild them with `python scripts/build_publication_figures.py`.

This repository implements the research contract in
`docs/research-specification.md`. The central distinction is
between observations released to the predictor and outcomes released to online
calibration. Evaluation retains a separate originally-valid truth stream.

The registered FAR-GW, SUMO, CoRel-adaptation, STAEformer, STID, and DCRNN
calibration studies are complete. Remaining publication work is tracked in
`docs/progress.md` and `reports/publication-readiness.md`; the machine-readable
environment, hash, seed, evidence, and limitation record is in
`reports/reproducibility-manifest.json`. CPU smoke artifacts remain explicitly
marked `smoke_only`.

## Environment and checks

The current verified environment is Windows 11, Python 3.12.14, CPU-only PyTorch
2.14.0, and SUMO/TraCI 1.27.1. Create a virtual environment, install
`requirements.lock`, and then install this package in editable mode. On this
workspace:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q
```

## Implemented commands

```powershell
python -m ps1.cli audit-environment
python -m ps1.cli audit-data --dataset metr_la
python -m ps1.cli prepare --dataset metr_la --config configs/metr_la.yaml
python -m ps1.cli make-faults --dataset metr_la --scenario C3 --seed 101
python -m ps1.cli make-registered-faults
python -m ps1.cli sumo-build --config configs/sumo.yaml
python -m ps1.cli sumo-generate --config configs/smoke.yaml --seed 11 --family P0
python -m ps1.cli sumo-paper --config configs/sumo.yaml --workers 3
python -m ps1.cli prepare-sumo
powershell -ExecutionPolicy Bypass -File scripts/train_sumo_seeds.ps1
powershell -ExecutionPolicy Bypass -File scripts/evaluate_sumo_clean.ps1
powershell -ExecutionPolicy Bypass -File scripts/evaluate_sumo_static_faults.ps1
python -m ps1.cli sumo-live --config configs/smoke.yaml
python -m ps1.cli smoke --config configs/smoke.yaml
python -m ps1.cli report --registry configs/experiment_registry.yaml
python -m ps1.cli train-feasibility --dataset metr_la --seed 11
python -m ps1.cli train --dataset metr_la --seed 11
python -m ps1.cli cache-predictions --dataset metr_la --seed 11 --partition calibration
python -m ps1.cli cache-predictions --dataset metr_la --seed 11 --partition test
python -m ps1.cli cache-faulted --dataset metr_la --model-seed 11 --scenario C3 --fault-seed 101
python -m ps1.cli evaluate-online --dataset metr_la --model-seed 11 --point-scenario C3 --feedback-scenario C2 --fault-seed 101
python -m ps1.cli evaluate-online --dataset metr_la --model-seed 11 --point-scenario C3 --feedback-scenario C3 --fault-seed 101
python -m ps1.cli evaluate-clean --dataset metr_la --seed 11
python -m ps1.cli evaluate-controls --dataset metr_la
powershell -ExecutionPolicy Bypass -File scripts/train_staeformer_matrix.ps1
python scripts/run_staeformer_evaluation_matrix.py --device cuda
python scripts/summarize_backbone_transfer.py
python scripts/build_reproduction_manifest.py
python scripts/run_real_core_evaluation_matrix.py --device cuda
powershell -ExecutionPolicy Bypass -File scripts/train_stid_matrix.ps1
python scripts/run_stid_evaluation_matrix.py --device cuda
powershell -ExecutionPolicy Bypass -File scripts/train_dcrnn_matrix.ps1
python scripts/run_dcrnn_evaluation_matrix.py --device cuda
```

Some point-comparator and live-integration commands in the research contract
remain under implementation. The status-report command is artifact-backed
and refuses completed entries without existing evidence. See `docs/progress.md` and
`docs/limitations.md` for exact gate status. Large datasets and run artifacts are
ignored; manifests record their hashes, schema, provenance, seeds, and status.
