# Figure 2 v1 QA

## Figure contract
- Core conclusion: DHC yields paired improvements, responds to the content and ordering of distant history, and provides a smaller but visible gain when attached to TimeXer.
- Archetype: asymmetric quantitative grid with panel a as the hero panel.
- Backend: Python only (`matplotlib`, `pandas`, `numpy`).
- Final canvas: 7.2 × 4.45 inch (518.4 × 320.4 pt).

## Data and statistics
- Panel a: 24 task–seed pairs from the test set; all MSE and MAE improvements are positive.
- Panel b: 4 tasks × 6 conditions; intervention effects are validation-set results from one frozen checkpoint, seed 2021; no error bars are implied.
- Panel c: 12 task–seed pairs from the test set; MSE wins 10/12 and MAE wins 8/12.
- No inferential test, confidence interval or significance symbol is used.
- Source-data CSV contains 60 rows: a=24, b=24, c=12.

## Export QA
- SVG contains editable text nodes.
- PDF contains embedded Unicode TrueType fonts.
- PNG and TIFF are 4320 × 2670 pixels at 600 dpi.
- Visual inspection at publication aspect ratio found no clipping or overlapping labels.
- Warm colour is used only for intervention severity; dataset identity also uses marker shape for horizon and remains interpretable without relying on red/green contrast.

## Manuscript consistency issue to resolve before insertion
- Table 4 lists task-level improvements computed from cross-seed metric means. Averaging those eight displayed task-level improvements gives 8.932% MSE and 4.181% MAE.
- The current prose reports 8.828% and 4.162%, which match averaging seed-level paired improvement ratios instead.
- The figure avoids mixing these definitions by showing only individual task–seed pairs. Before submission, the manuscript should either relabel 8.828%/4.162% as seed-pair averages or replace them with 8.932%/4.181% as task-level macro averages.
