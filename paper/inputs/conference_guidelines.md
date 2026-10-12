# Conference Guidelines

* **Venue type:** Single-blind technical report / competition technical paper.
* **Page limit:** 12 pages of body text, excluding references and appendix.
* **Language:** Chinese body text with an English abstract. Compilation must use
  `xelatex` with the `ctexart` document class. Chinese fonts available on the
  build machine: SimSun, SimHei, Microsoft YaHei.
* **Columns:** Single column, A4, 11 pt.
* **Mandatory sections:** Abstract, Introduction, Related Work, Method,
  Experimental Setup, Results, Analysis, Negative Results, Discussion and
  Limitations, Conclusion. An appendix may follow the references on a fresh
  page.
* **Emphasis requirement (binding):** The paper must be **results-first**. The
  Method section is limited to the explicitly claimed contributions only; it
  must not describe routine engineering plumbing. All other process detail
  belongs in the appendix or is omitted.
* **Tables:** `booktabs` style (`\toprule`, `\midrule`, `\bottomrule`). All
  numeric values must be copied exactly from `experimental_log.md`. No
  hallucinated numbers.
* **Figures:** All figures must appear before the Conclusion. Use the exact
  filenames provided in `inputs/figures/`. One figure per float; do not merge.
  Captions must not contain the string "Figure N".
* **Citations:** Use `natbib` with `\citep{}`. Only keys present in `refs.bib`
  may be cited.
* **Submission deadline:** 2026-10-15. Literature cutoff date for any
  literature search: 2026-09-15.
* **Anonymity:** No author names, affiliations, e-mail addresses, institutional
  logos or supervisor names may appear anywhere in the document.
