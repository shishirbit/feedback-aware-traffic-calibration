# Manuscript revision audit - 27 September 2026

The single `main.tex` contains all sections, equations, two algorithms, 19 tables, eight figure captions and the bibliography. Tables and figures are placed near their explanatory text. External architecture and result figures are vector PDFs, with PNG/TIFF alternatives; no TikZ is used.

The reference revision replaces draft versions with published proceedings or journal articles, removes two unpublished feedback studies and adds older, intermediate and recent journal literature. All 26 entries are cited and checked against publisher/proceedings metadata and corroborating bibliographic or author records. Detailed verification is in reference-audit.md and reference-verification.json. Original published conference papers remain with explicit author authorization. No journal indexing status was invented.

The introduction, literature review, calibration comparison table and discussion were revised to match the published sources. Quantile-loss and interval-score foundations are now cited. Reviewed but unevaluated methods are labelled accordingly. The abstract remains 150 words; Introduction and Related Work remain 1,000 and 1,500 prose words. The final PDF is 35 pages. No unresolved citations/references or overfull boxes remain; the revised table and bibliography were visually checked. Numerical results and registered claim limits are unchanged.

## Abstract revision - 28 September 2026

The unstructured abstract was rewritten to state the problem, practical evaluation gap, objective, causal method, datasets, key registered outcomes and conditional conclusion in exactly 150 words. The 35-page PDF recompiles without unresolved references or overfull text. The first page was visually checked. No numerical results or broader claims were added.

## Abstract numerical-results correction - 28 September 2026

In response to the author, the 150-word abstract now reports the FAR-GW METR-LA C3 improvement versus rolling calibration (0.355 m/s, 95% paired interval 0.192-0.551, coverage 0.8985), the 12/16 backbone decisions, SUMO C4 coverage 0.8793 against its 0.88 floor, and the adverse FAR-GW PEMS-BAY C4 effect (rolling better by 0.119 m/s). TeX percent escaping and page fit were checked; the complete abstract appears in the compiled PDF. No experimental outcomes changed.

## Title revision - 28 September 2026

Current title: Feedback-aware calibration of traffic prediction intervals under delayed and missing sensor reports. The LaTeX title and PDF metadata were updated together; the title avoids implying causal-effect estimation. Numerical results, abstract and references were unchanged.
