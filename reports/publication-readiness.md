# Publication readiness audit

Initial audit generated from source commit `aaf5da2faf6c45fe87e5ae8a789a5188b7688feb`; updated September 26, 2026 from the current uncommitted reporting worktree. Historical test counts below are not a claim of a fresh full-suite run.

The repository is not yet submission-ready. The locked FAR-GW core C1/C2/C5, matched-rate independent-control, and one-factor C3/C4 stress matrices are now complete (168, 120, and 1,488 stages). The reporting update adds six stress figures and full paired-effect tables. The matched FAR-Cal archive-cap compute–quality benchmark is complete (18 timed replays); The full registered ablations are now complete (432 stages); any retained full live-study claim remains unfinished.

| Required deliverable | Status | Evidence or remaining work |
|---|---|---|
| Executable repository | Complete | 78 tests pass; registered training/evaluation commands and locked environments are present. |
| Exhaustive experiment registry | Partial | Core, matched controls and stress are complete; full ablations are complete. |
| SUMO package | Complete | 120 generated episodes, prepared manifest, causal replay, mechanism subset, recovery, and live equivalence smoke are present. |
| Results package | Partial | Core, stress, independent controls, CoRel and four-backbone evidence are complete; full ablations are complete. |
| Required figures | Generated (6/6) | Coverage-width, spatial heatmap, outage timeline, recovery and outage-duration figures are generated. The descriptive archive-cap compute-quality figure is generated. |
| Manuscript | Draft complete | Claims follow the 12/16 cross-backbone result and explicitly reject universal superiority. |
| Reproduction manifest | Complete | Environment, commit, hashes, seeds, configs, hardware inventory, evidence hashes, and limitations are machine-readable. |

## Supported claim boundary

The four-backbone matrix is complete: **12/16** registered dataset/scenario cells meet the superiority rule. All four backbones pass METR-LA C3/C4. FAR-GW and STID do not pass PEMS-BAY C3/C4, so universal backbone-independent superiority is unsupported.

## Required next experiments

1. Completed: registered real-data C1, C2, C5, matched independent controls, and one-factor stress matrix. See `real-stress-results.md` and its provenance audit.
2. Completed: full registered seven-component asymmetric FAR-GW ablations across METR-LA, PEMS-BAY and full SUMO C3/C4 sets. All 432 manifests and 5,328 chunks verified; see `full-registered-ablations.md` and figure 15.
3. Completed: matched archive-cap runtime benchmark and compute–quality figure. This benchmark concerns FAR-Cal resource sensitivity, not a cross-method runtime ranking.
4. Complete the full live gateway/forecaster study if a real-time claim is retained.

## Dataset distribution decision

Raw METR-LA and PEMS-BAY files remain excluded. The DCRNN software license does not establish rights for the benchmark data, and the inspected public records do not state an explicit license for the exact packages. Reproduction instructions therefore require users to fetch the data and verify hashes locally.

Machine-readable details are in `reports/reproducibility-manifest.json`.

The complete figure inventory, selection rules, and editorial placement are recorded in [publication-figure-suite.md](publication-figure-suite.md). The generated output manifest is `reports/figures/publication/figure-manifest.json`.

## Full ablation completion — September 27, 2026

All 432 stages completed. Among 42 exploratory paired intervals, 13 indicate worse interval score after removal, four indicate improved score, and 25 include zero. Removing target-age weighting improves score in METR-LA C4 and PEMS-BAY C3/C4; removing context matching improves PEMS-BAY C4. The largest degradation is removal of stale blending in PEMS-BAY C4: +9.441 mph interval score, 95% CI [8.410, 10.511], coverage 0.549. These findings do not establish that every component is universally beneficial. The runtime warning from averaging fully masked cells is expected for non-focus times; all reported estimates and CI bounds are finite.

The manuscript must use the author-supplied Interact/Chicago author-date template and the requirements in `docs/manuscript-submission-requirements.md`. Current journal-specific compliance, author declarations and a verified public repository URL remain pending.


## Manuscript and reproducibility release — 2026-09-27

The supplied Interact template manuscript now includes author metadata, confirmed none declarations, mathematical formulation, causal algorithm, detailed external architecture, nine tables and four SI figures. Journal instructions were verified directly. Public evidence/code/generated-SUMO repository: https://github.com/shishirbit/feedback-aware-traffic-calibration. Raw real benchmarks remain acquisition-and-hash owing to unresolved redistribution rights. Final author scientific review is required before submission; journal acceptance or Q1 suitability is not guaranteed.
