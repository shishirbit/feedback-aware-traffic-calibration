# Registered protocol implementation notes

The authoritative protocol is `docs/research-specification.md`.
This file records executable interpretations and never relaxes that specification.

- Dataset boundaries are selected on the original HDF timestamp rows at
  60/10/10/20. PEMS-BAY's single 65-minute gap is then expanded with 12 wholly
  invalid five-minute rows. Windows are assigned by target timestamps, not by the
  row indices after reindexing.
- Benchmark zero values are invalid because the pinned DCRNN implementation uses
  zero as `null_val`. SUMO zero speeds remain physical values; empty E1 intervals
  alone are invalid.
- C0-C5 schedules are created before test evaluation. Negative values appear only
  in evaluator-side schedule arrays as an infinity sentinel and are never exposed
  to a predictor. Consumer events contain only packets released at the current bin.
- All recovery windows contain 12 origins at or after restoration, require 100
  valid pairs, and require three consecutive passing window ends. Nonrecovery is
  represented as censored. The joint diagnostic compares interval score to 1.10
  times the paired immediate-feedback oracle, using a flagged epsilon only at zero.
- Real-data uncertainty uses paired moving blocks with a registered 24-hour primary
  block and 12/48-hour sensitivity. SUMO bootstraps complete paired episodes.
