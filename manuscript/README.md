Single-file manuscript source

main.tex contains all sections, two algorithms, all 19 tables, eight figure environments/captions and the complete bibliography. There are no input/include dependencies or separate section/table/caption .tex files. Figures are external vector PDFs in figures/. Compile main.tex with Tectonic or LaTeX; no BibTeX step is required. The supplied Interact class is retained.

Tables and figures are embedded after their first explanatory paragraph within the appropriate section, rather than collected at the end. Runtime material remains in the compute subsection. The user's explicit embedded-layout instruction supersedes the earlier separate-page submission layout. The compiled PDF has 33 pages. All visual/algorithm references resolve; no overfull boxes were reported. Numerical content and section prose lengths remain unchanged.

main.tex is the authoritative manuscript source. Older asset builders can regenerate figures but also emit obsolete helper .tex files; do not use them to reconstruct the manuscript. The prior modular source is preserved outside the active manuscript in artifacts/manuscript-qa/pre-single-file.zip.
