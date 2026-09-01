# Figure 2 v2 QA

## Figure contract

- Core conclusion: DHC consistently improves the RGSP backbone, matched distant-history content matters to the observed correction, and transfer to TimeXer is positive on average but not universal.
- Evidence hierarchy: panel a is the hero result; panel b is a compact input-content diagnostic; panel c is a bounded transfer check.
- Backend: Python only (`matplotlib`, `pandas`, `numpy`, project `.venv`).
- Final canvas: 7.2 × 4.55 inch (518.4 × 327.6 pt).
- Reference-informed grammar: aligned task rows, explicit zero baselines, shared metric encoding, one compact matrix, and a single asymmetric composite rather than three independent figures.

## Data and statistical semantics

- Panel a: 8 tasks × 3 paired test seeds = 24 rows; every displayed MSE and MAE improvement is positive.
- Small markers are raw paired seeds; large markers are arithmetic task means; horizontal segments are observed min--max ranges. They are not confidence intervals.
- Panel b: 4 tasks × 6 conditions = 24 validation rows from one frozen checkpoint, seed 2021. The mismatch intervention has the largest MSE increase in every task.
- Panel c: 4 tasks × 3 paired test seeds = 12 rows; MSE wins 10/12 and MAE wins 8/12. Negative transfer values remain visible.
- No p-value, confidence interval, significance marker, or causal claim is introduced.
- Source-data CSV contains 60 rows: a=24, b=24, c=12.

## Export QA

- SVG retains editable text (`svg.fonttype = none`).
- PDF is one 518.4 × 327.6 pt page with embedded Unicode TrueType fonts (`pdf.fonttype = 42`).
- PNG and LZW-compressed TIFF are 4320 × 2730 pixels at 600 dpi.
- Full-resolution Python PNG and the compiled double-column manuscript page were visually inspected; titles, notes, tick labels, panel labels, and the intervention matrix are not clipped or overlapping.
- The right-hand diagnostic panels were enlarged by replacing the five-column grid with a compact two-column asymmetric layout; all essential labels remain at or above approximately 6 pt at final size.
- The palette avoids red--green contrast. Metric identity is redundant through both colour and marker shape; warm sequential colour is reserved for intervention severity.

## Reproducibility

Run from the repository root:

```bash
.venv/bin/python scripts/plot_timerole_figure2_evidence_v2.py
```

The script reuses the validated loaders and integrity checks in `scripts/plot_timerole_figure2_evidence.py` and writes a separate v2 bundle, leaving v1 unchanged.

## Manuscript consistency item retained from v1

- Table 4's task-level macro averages are 8.932% MSE and 4.181% MAE, whereas the current prose values 8.828% and 4.162% correspond to averaging seed-level improvement ratios. The figure avoids conflating these definitions by displaying raw paired seeds and task means without reporting one overall average.
