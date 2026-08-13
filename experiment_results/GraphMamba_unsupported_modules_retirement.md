# GraphMamba unsupported-module retirement

Date: 2026-08-13

## Decision

The active `models/GraphMamba.py` retains only the periodic multi-resolution
path that passed the validation gate:

- local patches `(4, 2)` and complete-period patches `(24, 12)`;
- one shared Mamba encoder called independently for the two token grids;
- the descriptor-conditioned scale adapter;
- post-scan feature concatenation and the existing graph mixer/head.

The following experimental paths were removed from the active model and CLI:

- physical-time-overlap state exchange;
- confidence-modulated adapter V2;
- confidence-aware output routing;
- cross-spectral LagGraph conditioning, including its periodic integration.
- global period-normalized Mamba delta V3 (learned and fixed-physical forms).

## Evidence retained

No experiment result was deleted. Design details, configurations, metrics, and
diagnostics remain in:

- `GraphMamba_periodic_multiresolution_v1_design.md`
- `GraphMamba_periodic_multiresolution_v1_validation.md`
- `GraphMamba_periodic_multiresolution_v2_design.md`
- `GraphMamba_periodic_multiresolution_v2_validation.md`
- `GraphMamba_periodic_laggraph_v1_design.md`
- `GraphMamba_periodic_laggraph_v1_validation.md`
- `GraphMamba_LagGraph_validation_gate.md`
- `GraphMamba_periodic_delta_v3_design.md`
- `GraphMamba_periodic_delta_v3_validation.md`
- `GraphMamba_periodic_delta_v3_code_archive.md`
- `scripts/diagnose_graphmamba_periodic_delta_v3.py`
- `scripts/diagnose_graphmamba_periodic_v1_structure.py`
- `scripts/diagnose_graphmamba_periodic_v1_checkpoint.py`
- `logs/graphmamba_periodic_v1_validation/`
- `logs/graphmamba_periodic_v2_validation/`
- `logs/graphmamba_periodic_laggraph_validation/`
- `logs/graphmamba_periodic_delta_v3/`
- `logs/graphmamba_periodic_delta_v3_validation/`

The diagnostic scripts are retained as historical specifications and may refer
to APIs that no longer exist in the active model. They are not current smoke
tests.

## Restoration rule

A retired path may be reintroduced later as a new opt-in revision when there is
a materially different hypothesis and a preregistered validation comparison.
The retained reports and diagnostics provide the behavior, equations,
configuration, and prior negative results needed for reconstruction. It should
not silently return to the default backbone.
