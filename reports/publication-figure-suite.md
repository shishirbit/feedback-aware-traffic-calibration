# Publication figure suite

This registry defines the compact main-paper and supplementary figure set for an IEEE T-ITS submission. It is broader than the six minimum figure families in the research specification. Every generated figure has a vector PDF, a 600-dpi PNG, and a CSV source table in `reports/figures/publication/`. The machine-readable manifest records captions and SHA-256 hashes of all source artifacts.

## Generated now

| ID | Figure | Placement | Evidence role |
|---|---|---|---|
| Fig. 1 | System and information flow | Main | Separates predictor input, immutable forecasts, replay, and delayed feedback. |
| Fig. 2 | C0–C5 scenario design matrix | Main | Makes channel interventions and controls explicit. |
| Fig. 3 | Clean point accuracy | Main | Reports MAE/RMSE by horizon with seed variability. |
| Fig. 4 | Clean calibration quality | Main | Provides coverage–width and reliability views. |
| Fig. 5 | Cross-backbone transfer | Main | Shows paired effects and uncertainty for all 16 registered cells. |
| Fig. 6 | Representative outage timeline | Main | Uses a prespecified median baseline-score event, avoiding favorable selection. |
| Fig. 7 | Spatial speed/error heatmap | Main | Shows where and when the selected outage affects accuracy. |
| Fig. 8 | Recovery curves | Main | Shows post-restoration coverage and score ratio, with censoring stated. |
| Fig. 9 | Component ablation | Main | Reports component-removal effects with uncertainty. |
| Fig. 10 | Mechanism isolation | Supplement | Separates delayed-feedback and predictor-input mechanisms. |
| Fig. 14 | All evaluated benchmark comparisons | Main | Four-panel forest plot: Rolling/ACI across four backbones and CoRel on FAR-GW, with 95% paired confidence intervals and the proposed FAR-Cal zero-reference. |

## Completed extension evidence

| ID | Figure | Blocking evidence |
|---|---|---|
| Fig. 11 | Core C1/C2/C5 extension | Completed 168-stage matrix; C0 and C3/C4 remain in the clean and transfer figures. |
| Fig. 12 | Interval score versus outage duration | Completed one-factor matrix; generated with PDF, 600-dpi PNG and CSV. Five additional stress-family figures accompany it. |
| Fig. 13 | Compute–quality trade-off | Completed 18-replay matched archive-cap benchmark; PDF, PNG and CSV generated. Single-seed descriptive sensitivity; interrupted evaluation-session runtimes are excluded. |

Pending figures are entries in the manifest rather than blank or synthetic plots. Run `python scripts/build_publication_figures.py` after their dependencies complete to rebuild the suite.

## Editorial use

For a ten-page regular paper, use Figs. 1–9 as composite main-paper figures only if the page budget permits. Fig. 10 and detailed per-seed/per-horizon panels belong in supplementary material. Fig. 11 should replace redundant result panels once complete. Figures 12–13 should be added only after their registered experiments produce comparable evidence.

The figures use color plus markers, line styles, or text so scientific distinctions survive grayscale reproduction. Two-column figures are 7.16 inches wide; PDFs contain vector content and PNG derivatives are rendered at 600 dpi.
