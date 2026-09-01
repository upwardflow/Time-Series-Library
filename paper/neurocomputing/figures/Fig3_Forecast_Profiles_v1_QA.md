# Figure 3 QA

## Figure contract

- Conclusion: TimeRole's qualitative behavior can be inspected on identical real targets at both H=96 and H=720, with a strict same-origin comparison against an internal control (RGSP-96) and an external baseline (DLinear).
- Selection protocol: ground truth only; OT channel; origin nearest median total variation among all valid H=720 test origins; identical origin reused across methods and horizons.
- Backend: Python only (`matplotlib`, `numpy`, `pandas`, project `.venv`).

## Data integrity

- Eight compact checkpoint exports were validated: 2 datasets × 2 horizons × 2 methods.
- For each panel, methods share identical observed context, target and origin.
- In each row, the H=96 target equals the first 96 values of the H=720 target.
- Source-data CSV contains 5,280 rows; window-metric CSV contains 8 rows; neither has missing values.
- Window annotations report original-scale MAE for the displayed origin only, not full-test aggregate performance.

## Export QA

- PDF: one 518.4 × 313.2 pt page with embedded Unicode TrueType fonts.
- PNG/TIFF: 4320 × 2610 pixels at 600 dpi.
- SVG: editable text retained (`svg.fonttype = none`; 66 text elements).
- Full-resolution PNG visually inspected after the final render; the forecast boundary, labels and all curves are legible without annotation overlap.

## Reproducibility

```bash
.venv/bin/python scripts/collect_timerole_forecast_profiles.py
.venv/bin/python scripts/plot_timerole_forecast_profiles.py
```
