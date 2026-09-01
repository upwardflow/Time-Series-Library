# Figure 3 model-curve candidate guide

## What is included

- Five datasets: ETTh1, ETTh2, ETTm1, ETTm2, and Weather.
- Four forecast horizons per dataset: 96, 192, 336, and 720.
- Eight model panels per candidate: TimeRole, S-Mamba, iTransformer, TimeMixer, MSGNet, PatchTST, TimesNet, and DLinear.
- One 20-page candidate book, 20 detailed PDF/SVG/PNG figures, five dataset overview pages, and a source-data CSV.

## Fair-selection protocol

For each dataset, the selected forecast origin is the valid H=720 test window whose ground-truth target total variation is nearest the dataset median. The same origin is reused for H=96, 192, 336, and 720, and the shorter targets are verified as exact prefixes of the H=720 target. No model error is used to choose the window.

All panels within a candidate use the same 96-step observed input, ground truth, origin, target channel, standardized scale, and y-axis limits. The blue and orange curves coincide over the input region and separate only after the dashed forecast boundary. Curves come from seed-2021 frozen checkpoints corresponding to the manuscript's main comparison. The ten TimeMixer cells identified in the manuscript as stability reruns use their validated learning-rate-1e-4 checkpoints.

## Suggested first-look candidates

| Dataset | Suggested horizon | Reason for first look |
|---|---:|---|
| ETTh1 | 96 | Clear local peaks and visible amplitude/phase differences without long-horizon crowding. |
| ETTh2 | 96 | Strong periodic structure and readily distinguishable tracking behavior. |
| ETTm1 | 192 | Contains repeated medium-scale cycles while retaining readable line density. |
| ETTm2 | 336 | Multiple cycles remain visible while the input-to-forecast transition is easy to inspect. |
| Weather | 192 | Shows a pronounced regime transition and recovery over a readable horizon. |

These are layout/readability suggestions, not performance-selected samples. Any of the 20 detailed candidates can be used.

## Interpretation boundary

The panels intentionally contain no MAE/RMSE annotations. They are qualitative, standardized-scale case studies for inspecting continuity, phase, and shape; full-test metrics and multi-seed statistics remain in the manuscript tables.
