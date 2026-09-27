# Acceptance gates and continuation

Goal active. Initial repository construction started 2026-09-08.

| Gate | Status | Remaining |
|---|---|---|
| A environment/data | substantially complete | both benchmarks acquired/audited/prepared and all 78 current tests pass under Python 3.12.14; the earlier 73-test suite passed under Python 3.11.9; dataset rights verification remains |
| B causal engine | substantially complete | causal/replay/config/data tests pass; SUMO online evaluation resets calibration, pending feedback, and adaptive state for every held-out episode |
| C predictors | substantially complete | all FAR-GW, STAEformer, STID, and DCRNN registered runs are complete; remaining work is final bundle review |
| D calibration | in progress | asymmetric FAR-Cal met the full rule on SUMO C3 and METR-LA C3/C4; it failed PEMS-BAY C3/C4, while SUMO C4 improved score but missed the coverage floor by 0.0007 |
| E real experiments | substantially complete | Core C1/C2/C5, matched independent controls and 1,488-stage one-factor stress are complete alongside prior C0/C3/C4 audits; full ablations remain |
| F SUMO generation | complete | generated and hash-validated all 120 episodes; prepared 11,520 rows with one-hour warm-up, episode-safe causal resets, and 3,660/915/915/1,830 train/tune/calibration/test windows |
| G SUMO replay/live | in progress | three-bin TraCI/E1 equivalence and current integration reruns pass; gateway/forecaster live path and physical-event pairing remain |
| H comparisons | substantially complete | rolling/ACI, FAR-Cal, CoRel, and complete STAEformer/STID/DCRNN 114-stage matrices are complete |
| I research report | substantially complete | Core/control/stress tables, manuscript extension, all six required figure families and the 18-replay descriptive compute-quality study are complete; submission editing and full ablations remain |

Preliminary artifact-backed results are available, but the registered replication matrix is incomplete. Do not mark the goal complete at an intermediate gate.
Recommend pausing only when useful implementation/review work is exhausted and the remaining work is an unattended run or requires external input.

## Verified initial checkpoint

`python -m pytest -q`: 78 current tests pass under Python 3.12.14. The earlier
73-test suite also passed under Python 3.11.9. Coverage includes SUMO integration,
deterministic paper-plan validation, episode-boundary checks, exact-array FAR-Cal
equivalence, and the additional backbone adapters; pytest
collection is scoped to `tests/`. The project-local `.venv` uses
Python 3.12.14 and CPU-only PyTorch 2.14.0, with exact versions in
`requirements.lock`; the Python 3.11.9 environment is recorded in
`requirements-py311.lock`. The RTX 5060 is visible to the driver but this PyTorch build
does not expose CUDA, so paper-scale GPU training is not yet configured.

Implemented: release-event validation, producer gateway, causal slot filling,
immutable forecasts, separate per-forecast feedback deduplication, exact FAR-Cal
mixture, block ESS, frozen order statistic, training-only statistics, split window
bounds, masked interval metrics, last-available control, consumer replay, and
trusted local engine checkpoint restoration.

The first complete CPU smoke used three independent three-hour P0 SUMO episodes.
Its 20-node FAR-GW ran two epochs/eight optimizer steps, produced finite loss,
initialized 3,120 residuals, and evaluated 3,120 valid targets. Artifacts are at
`artifacts/smoke-66538f1f9de7`; all metrics are marked `smoke_only` and are not
paper evidence.

Both real benchmarks are acquired, audited, and prepared. All registered FAR-GW
trainings and the complete asymmetric C3/C4 transfer audits are finished. SUMO
generation and fresh asymmetric validation are complete. Focused primary-source
checks cover ACI, MAS-Mamba, Post-Training ACP, and the spatiotemporal hypergraph
method. The direct CoRel adaptation and all four frozen-backbone matrices are
complete. The final bundle review and redistribution-rights audit are now explicit
in `reports/publication-readiness.md` and the machine-readable reproduction
manifest. The real-data C1/C2/C5, matched independent controls and one-factor
stress matrices are complete. The reporting update adds paired stress tables,
six stress figures, a core-extension figure, and an audit of 1,296 manifests
and 26,784 prediction/interval chunk hashes. The 18-replay archive-cap benchmark
and its compute-quality figure are complete, with six exact archived-quality
reproductions. Next: full ablations, submission editing, and any retained full
live-study claim.

## Full registered ablations — 2026-09-27

Locked configs/full_registered_ablations.json and launched scripts/run_full_registered_ablations.py: seven asymmetric-method ablations across METR-LA, PEMS-BAY and complete SUMO C3/C4 sets, model seeds 11/22/33 and fault seeds 101/202/303. Total 432 stages (54 verified full references plus 378 ablations). No new training or point-cache generation. Eleven targeted tests pass; real baseline metrics and every saved array match exactly, and a SUMO baseline episode also matches exactly. Completion will generate exploratory paired intervals, a result report, and figure 15; results are pending.

## Full ablations completed — 2026-09-27

Verified 432 completed manifests and 5,328 chunk hashes. Paired summaries and figure 15 generated and visually checked. Exploratory classification: 13 removals worsen score, four improve it and 25 are inconclusive. Both favorable and adverse findings must appear in the manuscript; no universal component-benefit claim. Remaining submission preparation includes author-supplied template compliance, verified journal instructions, public reproducibility release and author declarations.


## Manuscript and reproducibility release — 2026-09-27

The supplied Interact template manuscript now includes author metadata, confirmed none declarations, mathematical formulation, causal algorithm, detailed external architecture, nine tables and four SI figures. Journal instructions were verified directly. Public evidence/code/generated-SUMO repository: https://github.com/shishirbit/feedback-aware-traffic-calibration. Raw real benchmarks remain acquisition-and-hash owing to unresolved redistribution rights. Final author scientific review is required before submission; journal acceptance or Q1 suitability is not guaranteed.
