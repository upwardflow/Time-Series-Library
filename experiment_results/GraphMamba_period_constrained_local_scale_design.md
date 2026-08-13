# GraphMamba period-constrained local-scale design

Date: 2026-08-13

## Objective

Replace the inherited local patch `(4,2)` with a training-evidence derivation
that remains commensurate with the independently detected dominant period.

## Primary estimator

Use exactly the same training-only standardized moving-average residual as the
period diagnostic. For variable `i` and lag `l`, compute the overlap-normalized
autocorrelation

`rho_i(l) = <x_i[l:], x_i[:-l]> / (||x_i[l:]|| ||x_i[:-l]||)`.

Aggregate variables by the median `rho(l)`. Let the short-lag correlation
length be the first `l >= 2` and `l <= floor(P/2)` satisfying

`rho(l) <= rho(1) / e`.

If no crossing occurs, use `floor(P/2)` and mark the fallback. The `1/e`
threshold is fixed before inspecting ETTh1/ETTh2 local outputs.

## Period constraint

Construct the proper-divisor bank

`D(P) = {d : 2 <= d <= P/2 and P mod d = 0}`.

Choose the divisor nearest to the raw correlation length in log distance,
breaking exact ties toward the smaller divisor. Set

`local_patch = selected_divisor`

and

`local_stride = max(1, floor(local_patch/2))`.

For prime periods with no proper divisor, the periodic local-scale route is
declared unavailable rather than inventing a non-constrained patch.

## Stability diagnostics

- Repeat the primary derivation in four chronological training blocks.
- Report the fraction of blocks selecting the full-training result.
- Repeat full-training derivation at fixed relative thresholds `0.25`, `1/e`,
  and `0.5`.

These diagnostics measure sensitivity; they do not alter the primary selected
scale after outputs are observed.

## Integration gate

Model integration is allowed only if:

- both ETTh1 and ETTh2 produce a valid divisor-constrained scale;
- at least three of four chronological blocks agree with the full-training
  selected scale on each dataset; and
- the fixed-threshold sensitivity does not span more than adjacent divisor
  candidates.

If the rule selects the existing `(4,2)`, the result upgrades its provenance
but does not create a new architecture. A validation rerun is then a
reproducibility check, not evidence of a new gain. If it selects a different
stable common scale, compare it strictly against `(4,2)` on validation only.
