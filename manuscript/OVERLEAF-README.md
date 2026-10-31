# Springer Nature manuscript and Overleaf package

Import `springer-overleaf.zip` as a new Overleaf project. Select `main.tex` as the main document and **pdfLaTeX** as the compiler, then recompile. For an existing project, replace the old source and support files and use Recompile from scratch.

The package uses the author-supplied Springer Nature `sn-jnl` template (version 3.1, December 2024), with the numbered `sn-mathphys-num` style. It contains exactly one TeX source. All sections, equations, two algorithms, nineteen tables, eight figure environments and the formatted bibliography are included in `main.tex`. The figures are separate vector PDFs; no TikZ is used.

The 26-entry bibliography was generated using the supplied Springer bibliography style, then embedded in `main.tex`. No BibTeX step is needed. `references.bib` is included as editable bibliographic metadata. `highlights.txt` is supplied as a separate optional item.

Both authors share the Department of Computer Science and Engineering, Manipal University Jaipur affiliation. Satyabrata Roy is the corresponding author. Figures and tables are constrained to their nearby text and precede the references. The scientific body and numerical results were retained. The declarations use the Springer backmatter layout.

Validated with MiKTeX pdfLaTeX 26.5: 40 pages, eight figures, nineteen tables, two algorithms and 26 numbered references; no unresolved references or overfull boxes. The older Results in Engineering ZIP is an archive of the previous journal format.
