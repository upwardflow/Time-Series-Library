# Figure 3 candidate QA

## Data and alignment

- Formal model–dataset–horizon coverage: 160/160 exports.
- Each dataset–horizon candidate contains all eight models.
- Ground truth is numerically identical across models within every candidate.
- The 96-step observed input is numerically identical across models within every candidate.
- H=96, 192, and 336 targets are exact prefixes of the corresponding H=720 target for every dataset.
- All plotted standardized values are finite.
- Every export remains in normalized model space (`inverse_transformed = false`).
- Window selection uses ground truth only and is independent of model errors.

## Source provenance

- 150 exports use the formal seed-2021 command/checkpoint records directly.
- Eight ETTm2/Weather TimeMixer exports use the validated learning-rate-1e-4 stability-repair checkpoints.
- Two ETTm1 TimeMixer exports (H=96 and H=720) use the validated learning-rate-1e-4 repair checkpoints.
- The first-pass plots produced from obsolete unstable TimeMixer checkpoints were rejected and overwritten before delivery.

## Visual and export QA

- 20 detailed PDF, SVG, and PNG candidates generated.
- Five dataset overview PDF/PNG pages generated.
- Candidate book contains 20 pages.
- Detailed figure size is approximately 7.07 × 3.73 inches, suitable for double-column review.
- PDF fonts are embedded TrueType; SVG text remains editable.
- Shared axes, titles, legends, and forecast-boundary markers were visually inspected with no clipping.
- No MAE/RMSE annotation or metric subplot is present.
- Blue ground truth and orange prediction encoding is identical for every model; TimeRole receives no special color treatment.

## Interpretation

- Curves show one deterministic seed-2021 test window per dataset/horizon.
- Curves include 96 observed input steps followed by the requested forecast horizon on the standardized scale.
- Full-test and multi-seed claims remain supported by the manuscript tables, not by these representative curves.
