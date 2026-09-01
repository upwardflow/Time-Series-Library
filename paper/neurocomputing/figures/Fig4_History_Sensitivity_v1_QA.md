# Figure 4 QA

## Figure contract

- Conclusion: longer distant coverage helps long-horizon forecasting without requiring a longer high-resolution recent path; pooling compression is stable over the tested range.
- Evidence: finalized test results, three independently trained seeds per task-setting cell.
- Backend: Python only (`matplotlib`, `pandas`, `numpy`, project `.venv`).
- Statistical semantics: small points are dataset-level three-seed means and connected points are arithmetic macro means; no confidence interval or inferential test is claimed.

## Data integrity

- History-length sweep: ETTm1, ETTm2 and Weather; H=96/720; L=192/336/720/960; three seeds per cell.
- Recent/pooling sweeps: ETTm1 and ETTm2; H=96/720; recent=48/96/192 or pool=8/16/24; three seeds per cell.
- Source-data CSV contains 48 complete task-setting rows; macro-summary CSV contains 20 rows.
- H=720 macro MSE changes reproduce the manuscript values: −2.314% at L=720 and −3.566% at L=960 relative to L=336.

## Export QA

- PDF: one tightly cropped 515.657 × 193.615 pt page with embedded Unicode TrueType fonts.
- PNG/TIFF: 4297 × 1613 pixels at 600 dpi.
- SVG: editable text retained (`svg.fonttype = none`; 32 text elements).
- Full-resolution PNG visually inspected; shared scale, zero line, standard-setting bands, legend and panel labels are legible and unclipped.

## Reproducibility

```bash
.venv/bin/python scripts/plot_timerole_sensitivity.py
```
