# Empirical summary and supported claims

## Main finding

Asymmetric FAR-Cal improves interval score under correlated feedback impairment in fresh SUMO C3 episodes and in METR-LA C3/C4 transfer audits on all four tested backbones. STAEformer and DCRNN also meet the rule on PEMS-BAY C3/C4, while FAR-GW and STID do not. SUMO C4 misses the registered coverage floor narrowly. The evidence supports a conditional method contribution and multi-backbone transfer, not universal superiority.

| Dataset and condition | Rolling − method (95% CI) | ACI − method (95% CI) | Coverage | Registered rule |
|---|---:|---:|---:|---:|
| SUMO C3, fresh validation | 0.5947 [0.2356, 1.0065] | 0.5788 [0.2319, 0.9544] | 0.8890 | Met |
| SUMO C4, scenario transfer | 1.0285 [0.6083, 1.5249] | 0.9435 [0.5728, 1.3726] | 0.8793 | Not met |
| METR-LA C3, 3 × 3 audit | 0.7932 [0.4293, 1.2330] | 0.7723 [0.4141, 1.2039] | 0.8985 | Met |
| METR-LA C4, 3 × 3 audit | 0.9507 [0.5331, 1.4261] | 0.9512 [0.5656, 1.3808] | 0.8958 | Met |
| PEMS-BAY C3, 3 × 3 audit | -0.0786 [-0.2260, 0.1046] | -0.0936 [-0.2487, 0.0951] | 0.9006 | Not met |
| PEMS-BAY C4, 3 × 3 audit | -0.2651 [-0.4350, -0.0955] | -0.2908 [-0.4839, -0.1051] | 0.9068 | Not met |

Positive differences favor asymmetric FAR-Cal. SUMO inference bootstraps independent episodes. Real-data audits use hourly fault-focused origins with one-day moving blocks stratified by fault seed and a capped local-pooling implementation.

Effect figure: `reports/figures/asymmetric-effects.png`; plotted values: `reports/figures/asymmetric-effects.csv`.

## Claim boundary

The defensible empirical claim is that signed-tail, feedback-aware calibration can improve interval efficiency under correlated outages in some networks while retaining near-nominal coverage. Performance is heterogeneous: PEMS-BAY provides a clear counterexample under C4. No broad superiority, universal validity, or state-of-the-art claim is supported.

The original symmetric FAR-Cal formulation failed its confirmatory tests. The asymmetric redesign was locked before its fresh SUMO validation. The real-data transfer audits used the locked design without outcome-driven retuning.

## Direct relational comparator

The complete PS-1 CoRel adaptation audit favors FAR-Cal in every real-data C3/C4
condition. CoRel-minus-FAR-Cal interval-score effects were 1.6960 [1.1594,
2.3049] for METR-LA C3, 2.1767 [1.3784, 3.1390] for METR-LA C4, 2.2547
[1.8212, 2.7181] for PEMS-BAY C3, and 2.2450 [1.7615, 2.7452] for PEMS-BAY
C4. This supports superiority to the direct relational adaptation, while the
rolling/ACI PEMS-BAY counterexample still rejects universal superiority.

## Cross-backbone synthesis

The registered rule was met in 12/16 backbone/dataset/condition cells: 2/4 with
FAR-GW, 4/4 with STAEformer, 2/4 with STID, and 4/4 with DCRNN. All four backbones
met the rule on METR-LA C3/C4. STAEformer and DCRNN met it on PEMS-BAY C3/C4. This establishes
multi-backbone transfer on METR-LA but rejects universal backbone-independent
superiority.
Full evidence is in `reports/cross-backbone-transfer-synthesis.md`.

## Remaining publication work

- Verify dataset redistribution rights.
- Complete final bundle review while preserving the FAR-GW PEMS-BAY and SUMO C4 negative findings.
