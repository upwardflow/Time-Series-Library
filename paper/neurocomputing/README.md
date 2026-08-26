# TimeRole — Neurocomputing submission project

This directory is a self-contained LaTeX starting point for submitting TimeRole
to *Neurocomputing* with Elsevier's numerical citation style.

## Files

- `main.tex`: review manuscript in Elsevier `preprint` format.
- `manuscript_zh.tex`: complete Chinese body converted from the source manuscript.
- `highlights.tex`: five highlights for separate upload; not included in the manuscript PDF.
- `cover_letter.tex`: editable cover-letter draft.
- `references.bib`: snapshot of `../TimeRole_references.bib`.
- `elsarticle.cls`: generated from the downloaded official Elsevier bundle.
- `elsarticle-num.bst`: official numerical bibliography style.
- `figures/`: manuscript artwork; currently contains the TimeRole architecture PDF.

The Chinese source manuscript remains at `../TimeRole_中文主稿.md`; its complete body
is included by `main.tex` through `manuscript_zh.tex`.

## Build

The Chinese draft requires XeLaTeX and a CJK font. Run from this directory:

```bash
xelatex main.tex
pdflatex cover_letter.tex
```

For a clean manuscript build with citations, use the full sequence:

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
pdflatex cover_letter.tex
```

The source currently uses the Noto Serif/Sans CJK SC fonts configured in `main.tex`.
The full sequence resolves the bibliography and cross-references.

## Before submission

1. Replace all author, affiliation, email, declaration, and availability placeholders.
2. Verify every numerical result against the final experiment tables.
3. Confirm that every highlight is at most 85 characters in the submission portal.
4. Compile until citations and cross-references are resolved, then inspect every page.
5. Follow the current Neurocomputing Guide for Authors and portal checklist.
6. If Editorial Manager requires a flat LaTeX upload, place `main.tex`, the `.bib`,
   `.bst`, `.cls`, and every referenced figure at the same directory level and update
   `\graphicspath`/figure paths accordingly.
