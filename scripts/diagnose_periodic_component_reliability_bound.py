#!/usr/bin/env python3
"""Validation-only upper bound for Periodic Component Reliability Fusion.

The accepted periodic GraphMamba checkpoint is frozen.  Component forecast
contributions and causal reliability observables are extracted in chronological
order, ridge corrections are fit on train, and only validation is evaluated.
The test split is never constructed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_provider.data_factory import data_provider
from scripts.diagnose_horizon_phase_relation_bound import load_frozen_model


OUTPUT_ROOT = ROOT / "logs" / "graphmamba_pcrf_diagnostic"
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
MODEL_IDS = {
    "D0": "accepted_forecast_no_correction",
    "D1": "static_component_recalibration",
    "D2": "cycle_consistency_conditioned_seasonal",
    "D3": "trend_roughness_conditioned_trend",
    "D4": "joint_component_reliability",
    "D4_perm": "joint_with_permuted_training_reliability",
}


def ordered_loader(args: object, split: str) -> DataLoader:
    dataset, _ = data_provider(args, split)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )


def reliability_observables(
    seasonal: torch.Tensor,
    trend: torch.Tensor,
    period: int,
) -> torch.Tensor:
    """Return [B, N, 2]: adjacent-cycle cosine and log trend roughness."""
    if seasonal.shape[1] < 2 * period:
        raise ValueError("PCRF reliability requires at least two complete periods")
    latest = seasonal[:, -period:]
    previous = seasonal[:, -2 * period : -period]
    latest = latest - latest.mean(dim=1, keepdim=True)
    previous = previous - previous.mean(dim=1, keepdim=True)
    cycle_cosine = (latest * previous).sum(dim=1) / torch.sqrt(
        latest.square().sum(dim=1) * previous.square().sum(dim=1) + 1e-8
    )

    first = trend[:, 1:] - trend[:, :-1]
    second = first[:, 1:] - first[:, :-1]
    roughness = second.square().mean(dim=1) / (first.square().mean(dim=1) + 1e-8)
    log_roughness = torch.log1p(roughness)
    return torch.stack((cycle_cosine, log_roughness), dim=-1)


def extract_split(model, args: object, split: str) -> dict[str, np.ndarray | float]:
    if not model.use_periodic_multiscale or not model.use_decomp:
        raise ValueError("PCRF diagnosis requires periodic multiscale and decomposition")
    components, reliabilities, residuals = [], [], []
    equivalence_max_abs = 0.0
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in ordered_loader(args, split):
            x = batch_x.float().cuda()
            y = batch_y[:, -args.pred_len:].float().cuda()
            means = x.mean(dim=1, keepdim=True).detach()
            centered = x - means
            stdev = torch.sqrt(
                torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            normalized = centered / stdev
            seasonal, trend = model.decomposition(normalized)
            trend_normalized = model.trend_projection(
                trend.permute(0, 2, 1)
            ).permute(0, 2, 1)
            seasonal_state = model._periodic_multiscale_states(
                seasonal.permute(0, 2, 1)
            )
            seasonal_normalized = model.head(seasonal_state)
            reconstructed = (seasonal_normalized + trend_normalized) * stdev + means

            decoder = torch.cat(
                (
                    batch_y[:, : args.label_len].float().cuda(),
                    torch.zeros_like(y),
                ),
                dim=1,
            )
            direct = model(
                x,
                batch_x_mark.float().cuda(),
                decoder,
                batch_y_mark.float().cuda(),
            )
            equivalence_max_abs = max(
                equivalence_max_abs,
                float((reconstructed - direct).abs().max()),
            )
            components.append(
                torch.stack(
                    (seasonal_normalized * stdev, trend_normalized * stdev),
                    dim=-1,
                ).cpu()
            )
            reliabilities.append(
                reliability_observables(
                    seasonal, trend, int(args.periodic_period)
                ).cpu()
            )
            residuals.append((y - direct).cpu())
    return {
        "component": torch.cat(components).numpy().astype(np.float64),
        "reliability": torch.cat(reliabilities).numpy().astype(np.float64),
        "residual": torch.cat(residuals).numpy().astype(np.float64),
        "equivalence_max_abs": equivalence_max_abs,
    }


def standardize_reliability(
    train: np.ndarray,
    val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    return (train - mean) / scale, (val - mean) / scale, mean[0], scale[0]


def design_matrix(
    kind: str,
    component: np.ndarray,
    reliability: np.ndarray,
) -> np.ndarray:
    seasonal = component[..., 0]
    trend = component[..., 1]
    cycle = reliability[:, None, :, 0]
    roughness = reliability[:, None, :, 1]
    features = [seasonal, trend]
    if kind in {"D2", "D4", "D4_perm"}:
        features.append(seasonal * cycle)
    if kind in {"D3", "D4", "D4_perm"}:
        features.append(trend * roughness)
    if kind in {"D4", "D4_perm"}:
        features.extend((seasonal * roughness, trend * cycle))
    return np.stack(np.broadcast_arrays(*features), axis=-1)


def fit_coefficients(
    design: np.ndarray,
    residual: np.ndarray,
    alpha: float,
) -> np.ndarray:
    coefficients = np.zeros((design.shape[2], design.shape[3]), dtype=np.float64)
    for variable in range(design.shape[2]):
        x = design[:, :, variable].reshape(-1, design.shape[3])
        y = residual[:, :, variable].reshape(-1)
        xtx = x.T @ x / len(x)
        xty = x.T @ y / len(x)
        coefficients[variable] = np.linalg.solve(
            xtx + alpha * np.eye(x.shape[1]), xty
        )
    return coefficients


def apply_coefficients(design: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.einsum("shvf,vf->shv", design, coefficients)


def normalize_design(
    train: np.ndarray,
    val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rms = np.sqrt(np.mean(train * train, axis=(0, 1), keepdims=True))
    rms[rms < 1e-8] = 1.0
    return train / rms, val / rms, rms[0, 0]


def select_alpha(
    design: np.ndarray,
    residual: np.ndarray,
) -> tuple[float, list[dict[str, float]]]:
    boundary = int(0.8 * design.shape[0])
    scores = []
    for alpha in RIDGE_GRID:
        coefficients = fit_coefficients(
            design[:boundary], residual[:boundary], alpha
        )
        correction = apply_coefficients(design[boundary:], coefficients)
        error = residual[boundary:] - correction
        scores.append({"alpha": alpha, "mse": float(np.mean(error * error))})
    selected = min(scores, key=lambda row: (row["mse"], row["alpha"]))["alpha"]
    return float(selected), scores


def evaluate(
    kind: str,
    train: dict,
    val: dict,
    train_reliability: np.ndarray,
    val_reliability: np.ndarray,
    seed: int,
) -> tuple[dict, np.ndarray]:
    if kind == "D0":
        error = val["residual"]
        return {
            "model": kind,
            "description": MODEL_IDS[kind],
            "selected_alpha": None,
            "mse": float(np.mean(error * error)),
            "mae": float(np.mean(np.abs(error))),
        }, error

    fitting_reliability = train_reliability
    if kind == "D4_perm":
        permutation = np.random.default_rng(seed).permutation(
            fitting_reliability.shape[0]
        )
        fitting_reliability = fitting_reliability[permutation]
    train_design = design_matrix(kind, train["component"], fitting_reliability)
    val_design = design_matrix(kind, val["component"], val_reliability)
    train_design, val_design, rms = normalize_design(train_design, val_design)
    alpha, scores = select_alpha(train_design, train["residual"])
    coefficients = fit_coefficients(train_design, train["residual"], alpha)
    correction = apply_coefficients(val_design, coefficients)
    error = val["residual"] - correction
    return {
        "model": kind,
        "description": MODEL_IDS[kind],
        "selected_alpha": alpha,
        "alpha_scores": scores,
        "design_rms": rms.tolist(),
        "coefficients": coefficients.tolist(),
        "mse": float(np.mean(error * error)),
        "mae": float(np.mean(np.abs(error))),
    }, error


def moving_block_ci(
    reference_error: np.ndarray,
    candidate_error: np.ndarray,
    block: int,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    reference = np.mean(reference_error * reference_error, axis=(1, 2))
    candidate = np.mean(candidate_error * candidate_error, axis=(1, 2))
    rng = np.random.default_rng(seed)
    draws = []
    while_count = len(reference)
    for _ in range(repetitions):
        indices = []
        while len(indices) < while_count:
            start = int(rng.integers(0, max(1, while_count - block + 1)))
            indices.extend(range(start, min(start + block, while_count)))
        selected = np.asarray(indices[:while_count])
        ref_mse = reference[selected].mean()
        cand_mse = candidate[selected].mean()
        draws.append(100.0 * (ref_mse - cand_mse) / ref_mse)
    low, high = np.percentile(draws, (2.5, 97.5))
    return {
        "block_origins": block,
        "repetitions": repetitions,
        "improvement_low_pct": float(low),
        "improvement_high_pct": float(high),
    }


def diagnose_dataset(args: argparse.Namespace, dataset: str) -> dict:
    model, model_args, checkpoint, record = load_frozen_model(dataset)
    if model_args.pred_len != args.pred_len:
        raise ValueError("Requested prediction length does not match checkpoint")
    print(f"[{dataset}] extracting ordered train components", flush=True)
    train = extract_split(model, model_args, "train")
    print(f"[{dataset}] extracting ordered validation components", flush=True)
    val = extract_split(model, model_args, "val")
    train_rel, val_rel, rel_mean, rel_scale = standardize_reliability(
        train["reliability"], val["reliability"]
    )

    results, errors = {}, {}
    for kind in MODEL_IDS:
        print(f"[{dataset}] fitting {kind}: {MODEL_IDS[kind]}", flush=True)
        results[kind], errors[kind] = evaluate(
            kind, train, val, train_rel, val_rel, args.seed
        )
    d0_mse = results["D0"]["mse"]
    d1_mse = results["D1"]["mse"]
    for kind in ("D1", "D2", "D3", "D4", "D4_perm"):
        results[kind]["improvement_over_D0_pct"] = (
            100.0 * (d0_mse - results[kind]["mse"]) / d0_mse
        )
    for kind in ("D2", "D3", "D4", "D4_perm"):
        results[kind]["improvement_over_D1_pct"] = (
            100.0 * (d1_mse - results[kind]["mse"]) / d1_mse
        )

    checkpoint_mse = float(record["best_mse"])
    reproduction_error = abs(d0_mse - checkpoint_mse) / checkpoint_mse
    if reproduction_error > 1e-5:
        raise RuntimeError(
            f"{dataset} D0 mismatch: diagnostic={d0_mse}, "
            f"checkpoint={checkpoint_mse}, relative_error={reproduction_error}"
        )
    return {
        "dataset": dataset,
        "scope": "ordered_train_fit_ordered_validation_evaluation_no_test",
        "checkpoint": str(checkpoint),
        "checkpoint_recorded_validation_mse": checkpoint_mse,
        "D0_reproduction_relative_error": reproduction_error,
        "reconstruction_equivalence_max_abs": max(
            train["equivalence_max_abs"], val["equivalence_max_abs"]
        ),
        "train_origins": int(train["residual"].shape[0]),
        "validation_origins": int(val["residual"].shape[0]),
        "period": int(model_args.periodic_period),
        "reliability_mean": rel_mean.tolist(),
        "reliability_scale": rel_scale.tolist(),
        "models": results,
        "D4_bootstrap_vs_D1": moving_block_ci(
            errors["D1"], errors["D4"], args.bootstrap_block,
            args.bootstrap_repetitions, args.seed,
        ),
    }


def gate(rows: list[dict]) -> dict:
    d1_d0 = [row["models"]["D1"]["improvement_over_D0_pct"] for row in rows]
    d4_d0 = [row["models"]["D4"]["improvement_over_D0_pct"] for row in rows]
    d4_d1 = [row["models"]["D4"]["improvement_over_D1_pct"] for row in rows]
    main_effects = [
        row["models"]["D4"]["mse"]
        < min(row["models"]["D2"]["mse"], row["models"]["D3"]["mse"])
        for row in rows
    ]
    permutation = [
        row["models"]["D4"]["improvement_over_D1_pct"] > 0
        and row["models"]["D4_perm"]["improvement_over_D1_pct"]
        <= 0.5 * row["models"]["D4"]["improvement_over_D1_pct"]
        for row in rows
    ]
    passed = (
        min(d1_d0) > 0
        and np.mean(d1_d0) >= 1.0
        and min(d4_d1) > 0
        and np.mean(d4_d1) >= 0.5
        and min(d4_d0) >= 1.0
        and all(main_effects)
        and all(
            row["D4_bootstrap_vs_D1"]["improvement_low_pct"] > 0
            for row in rows
        )
        and all(permutation)
    )
    return {
        "D1_over_D0_each_pct": d1_d0,
        "D1_over_D0_macro_pct": float(np.mean(d1_d0)),
        "D4_over_D0_each_pct": d4_d0,
        "D4_over_D1_each_pct": d4_d1,
        "D4_over_D1_macro_pct": float(np.mean(d4_d1)),
        "D4_better_than_D2_D3_all": all(main_effects),
        "permutation_removes_half_all": all(permutation),
        "implementation_gate_passed": bool(passed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=("ETTh1", "ETTh2"),
        default=("ETTh1", "ETTh2"),
    )
    parser.add_argument("--pred-len", type=int, default=192)
    parser.add_argument("--bootstrap-block", type=int, default=24)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in args.datasets:
        row = diagnose_dataset(args, dataset)
        rows.append(row)
        (args.output_dir / f"{dataset}_p{args.pred_len}.json").write_text(
            json.dumps(row, indent=2) + "\n", encoding="utf-8"
        )
    payload = {
        "experiment": "GraphMamba_PCRF_D0_D4_diagnostic_v0",
        "configuration": vars(args) | {"output_dir": str(args.output_dir)},
        "datasets": rows,
        "gate": gate(rows),
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["gate"], indent=2), flush=True)
    print(f"Saved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
