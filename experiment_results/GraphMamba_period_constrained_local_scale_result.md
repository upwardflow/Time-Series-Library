# GraphMamba period-constrained local-scale result

Date: 2026-08-13

## Result

The preregistered training-only estimator independently recovers
`local_patch=4` and `local_stride=2` on both ETTh1 and ETTh2. This changes the
scale's provenance from an inherited constant to a period-constrained,
data-derived quantity. It does not change the current model's numerical
architecture and therefore is not claimed as a new validation improvement.

| Dataset | Period | Raw correlation length | Selected local patch/stride | Four block patches | Agreement |
|---|---:|---:|---:|---|---:|
| ETTh1 | 24 | 4 | 4 / 2 | 4, 4, 4, 4 | 4/4 |
| ETTh2 | 24 | 4 | 4 / 2 | 6, 4, 4, 4 | 3/4 |

The aggregate median residual autocorrelations at lags 1--6 were:

- ETTh1: `0.6947, 0.5169, 0.3201, 0.1255, -0.0341, -0.1596`.
- ETTh2: `0.6678, 0.4821, 0.3194, 0.1585, 0.0401, -0.0675`.

The corresponding `rho(1)/e` thresholds are 0.2555 and 0.2457. In both
datasets, lag 4 is the first lag at or below that threshold. Since 4 is a
proper divisor of the detected period 24, the period projection leaves it at
4; 50% overlap gives stride 2.

## Stability gate

The fixed sensitivity thresholds select patches `(4, 4, 3)` for correlation
ratios `(0.25, 1/e, 0.5)` on both datasets. Patches 3 and 4 are adjacent in the
ordered period-divisor bank `(2, 3, 4, 6, 8, 12)`. Together with block
agreement of 4/4 and 3/4, all preregistered integration conditions pass.

## Integration

`run.py` now treats zero-valued periodic local patch/stride arguments as an
auto mode. For periodic GraphMamba it loads the dataset's training-derived
record, checks that its period matches `periodic_period`, and resolves the
actual patch and stride before constructing the experiment identifier. An
explicit positive patch remains supported; a zero stride then means half the
patch. `GraphMamba.py` enforces that the resolved or explicit local patch is a
proper divisor no larger than half the period, preventing an off-contract
periodic geometry from silently entering the model.

Evidence records:

- `logs/graphmamba_local_scale/ETTh1_local_scale.json`
- `logs/graphmamba_local_scale/ETTh2_local_scale.json`

Reproduction:

```bash
python scripts/derive_graphmamba_local_scale.py --dataset ETTh1
python scripts/derive_graphmamba_local_scale.py --dataset ETTh2
```

## Interpretation boundary

The supported claim is: the local scale is the shortest period-compatible
resolution matching the training residual's e-folding correlation length.
The unsupported claim would be that patch 4 itself is a new architecture or
that this derivation has produced an accuracy gain. A future novelty claim
requires making this rule adaptive across datasets or samples and validating
that adaptation against fixed-scale controls.
