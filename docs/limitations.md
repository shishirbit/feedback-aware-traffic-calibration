# Current limitations

- CPU smoke output is execution evidence only. It cannot support H1-H4 or any
  novelty, calibration, robustness, recovery, real-time, or generalization claim.
- The base PyTorch environment is CPU-only. An isolated Python 3.11 / PyTorch
  2.13.0+cu130 environment now uses the RTX 5060 GPU for STAEformer; all nine
  registered FAR-GW training runs remain complete.
- Python 3.12.14 passes the current 78-test suite. Python 3.11.9 passed the
  earlier 73-test suite and executed the GPU backbone matrices; its compatible
  transitive versions are recorded separately in `requirements-py311.lock`.
- Dataset files are publicly downloadable, but the DCRNN software's MIT license
  does not license the benchmark data. The official Caltrans PeMS terms include
  third-party-rights caveats, and the inspected Zenodo benchmark records do not
  display an explicit license for the exact packages. Large inputs remain ignored;
  reproduction requires users to fetch and hash them locally.
- The final publication audit is recorded in `reports/publication-readiness.md`
  and `reports/reproducibility-manifest.json`. The bundle is not submission-ready:
  real-data C1/C2/C5/stress rows, the full registered ablation matrix, and all six
  required figure families remain incomplete.
- A source-split CoRel reproduction, full live gateway integration, stress tests,
  and full-scale FAR-Cal replication remain incomplete. The six-run STAEformer,
  STID, and DCRNN training matrices and all three 114-stage C3/C4 evaluations are complete.
  The focused literature audit is bounded by the
  primary-source search scope and must not be read as proof that no closer work exists.
- The complete CoRel comparison is a PS-1 multihorizon feedback adaptation of
  the official architecture. It supports a direct comparator claim within this
  protocol, not a claim about CoRel's published source-split performance.
- The ACI comparator is a documented spatial arrival-batch adaptation. Missing or
  delayed outcomes violate the protocol behind the original guarantee, so its
  results will be empirical comparisons rather than validity claims.
- The tuning-locked FAR-Cal candidate was evaluated on the complete 3 model-seed
  by 3 fault-seed SUMO C3 matrix (30 held-out episodes per run). Its mean interval
  score was worse than rolling by 0.150 and ACI by 0.234; no run established
  FAR-Cal superiority. Test-driven retuning is prohibited.
- A post-test audit found that the original tuning code had used the later
  calibration archive. The corrected first-third tuning warm-up and remaining-
  two-thirds scoring protocol selected the identical candidate. No parameter was
  changed, but the timing and original protocol deviation must remain disclosed.
- On 30 newly generated independent episodes, regularized FAR-Cal had lower mean
  interval score than rolling by 0.052 and ACI by 0.038 with 0.891 coverage.
  Both 95% episode-bootstrap intervals crossed zero, so the registered strict
  superiority rule was not met.
- A preregistered powered confirmation on 160 additional fresh episodes reversed
  the small descriptive effects: rolling and ACI each scored about 0.027 better
  than regularized FAR-Cal, with both 95% intervals crossing zero. The evidence
  does not support symmetric FAR-Cal superiority on SUMO C3.
- The subsequently locked asymmetric FAR-Cal redesign met its registered rule on
  40 fresh SUMO C3 episodes: rolling-minus-method was 0.595 (95% episode-bootstrap
  interval 0.236 to 1.006), ACI-minus-method was 0.579 (0.232 to 0.954), and
  coverage was 0.889. This supports superiority for that validation setting only;
  cross-dataset and broader scenario evidence remains incomplete.
- A resource-bounded hourly C3 transfer audit over three model and three fault
  seeds met the same comparison rule on METR-LA, but not on PEMS-BAY. PEMS-BAY
  comparator-minus-method estimates were -0.079 and -0.094 and both confidence
  intervals crossed zero. Broad cross-dataset superiority is therefore unsupported. The
  transfer audit uses a capped local-pooling approximation and fault-focused cells,
  so it is supporting evidence rather than a replacement for the full matrix.
- Under C4 permanent feedback loss, asymmetric FAR-Cal beat rolling and ACI in
  METR-LA, but was significantly worse in PEMS-BAY (comparator-minus-method
  estimates -0.265 and -0.291, with both intervals below zero). In SUMO C4 it
  improved interval score significantly but achieved 0.8793 coverage, narrowly
  below the registered 0.88 floor. These results reject broad C4 superiority.
- STAEformer and DCRNN met the rule in all four registered cells. STID met the
  rule on METR-LA C3/C4 but not PEMS-BAY C3/C4. Across all four tested
  backbones, 12/16 cells met the rule and every backbone passed METR-LA C3/C4.
  PEMS-BAY still separates the backbones: STAEformer and DCRNN passed, while
  FAR-GW and STID did not. Universal backbone-independent superiority remains unsupported.
- FAR-Cal is an empirical adaptive interval method. The implementation does not
  assert split-conformal finite-sample validity for weighted pooling or dependent,
  selectively observed test streams.
