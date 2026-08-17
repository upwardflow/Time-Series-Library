#!/usr/bin/env python3
"""Validation-only upper bound for horizon--phase conditioned relations.

The script keeps a frozen accepted periodic GraphMamba checkpoint, extracts its
normalized forecast residuals, and tests whether cross-variable history
summaries correct those residuals better when their ridge coefficients are
conditioned on forecast-distance and known future phase. It never reads test.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_provider.data_factory import data_provider
from models.GraphMamba import Model
from scripts.diagnose_graphmamba_representations import command_args


RECORD_ROOT = ROOT / "logs" / "graphmamba_periodic_v1_validation" / "validation"
OUTPUT_ROOT = ROOT / "logs" / "graphmamba_hpmrg_diagnostic"
MODEL_IDS = {
    "D0": "no_cross_correction",
    "D1": "shared_relation",
    "D2": "horizon_conditioned",
    "D3": "phase_conditioned",
    "D4": "horizon_phase_interaction",
    "D4_perm": "interaction_permuted_training_phase",
}
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def locate_checkpoint(dataset: str) -> tuple[dict, Path]:
    suffix = "h1" if dataset == "ETTh1" else "h2"
    record_path = RECORD_ROOT / f"pv1a_{suffix}_s21.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    matches = list((ROOT / "checkpoints").glob(
        f"*{record['candidate']}*/checkpoint.pth"
    ))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one checkpoint for {record['candidate']}, got {matches}"
        )
    return record, matches[0]


def load_frozen_model(dataset: str) -> tuple[Model, object, Path, dict]:
    record, checkpoint = locate_checkpoint(dataset)
    args = command_args(record["command"])
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    model = Model(args).cuda().eval()
    incompatible = model.load_state_dict(
        torch.load(checkpoint, map_location="cuda", weights_only=True),
        strict=False,
    )
    allowed_unexpected = (
        "periodic_exchange_", "periodic_router_", "periodic_confidence_",
        "local_from_period_alignment", "period_from_local_alignment",
    )
    unexpected = [
        key for key in incompatible.unexpected_keys
        if not key.startswith(allowed_unexpected)
    ]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch; missing={incompatible.missing_keys}, "
            f"unexpected={unexpected}"
        )
    return model, args, checkpoint, record


def history_features(x: torch.Tensor, period: int) -> torch.Tensor:
    """Four interpretable summaries per source variable, [B, N, 4]."""
    means = x.mean(dim=1, keepdim=True)
    scale = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
    normalized = (x - means) / scale
    local = min(4, normalized.shape[1])
    period = min(period, normalized.shape[1])
    return torch.stack(
        (
            normalized[:, -1],
            normalized[:, -local:].mean(dim=1),
            normalized[:, -period:].mean(dim=1),
            normalized[:, -1] - normalized[:, -period],
        ),
        dim=-1,
    )


def phase_bins_from_marks(marks: torch.Tensor, freq: str, bins: int) -> torch.Tensor:
    """Map known future Time-Series-Library marks to daily phase bins."""
    if freq.lower() in {"h", "1h"}:
        hour = torch.round((marks[..., 0] + 0.5) * 23.0)
        fraction = hour / 24.0
    elif freq.lower() in {"t", "min", "15min"}:
        minute = torch.round((marks[..., 0] + 0.5) * 59.0)
        hour = torch.round((marks[..., 1] + 0.5) * 23.0)
        fraction = (hour + minute / 60.0) / 24.0
    else:
        raise ValueError(f"No explicit daily phase decoder for frequency {freq!r}")
    return torch.floor(torch.remainder(fraction, 1.0) * bins).long().clamp_max(bins - 1)


def extract_split(
    model: Model,
    args: object,
    split: str,
    phase_bins: int,
) -> dict[str, np.ndarray]:
    _, loader = data_provider(args, split)
    features, residuals, phases = [], [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            x = batch_x.float().cuda()
            y = batch_y[:, -args.pred_len:].float().cuda()
            decoder = torch.cat(
                (
                    batch_y[:, : args.label_len].float().cuda(),
                    torch.zeros_like(y),
                ),
                dim=1,
            )
            forecast = model(
                x,
                batch_x_mark.float().cuda(),
                decoder,
                batch_y_mark.float().cuda(),
            )
            features.append(history_features(x, args.periodic_period).cpu())
            # The data loader has already standardized every channel with
            # training statistics. Dividing once more by a per-window scale can
            # explode nearly constant windows and invalidate the residual bound.
            residuals.append((y - forecast).cpu())
            phases.append(
                phase_bins_from_marks(
                    batch_y_mark[:, -args.pred_len:].float(),
                    args.freq,
                    phase_bins,
                )
            )
    return {
        "features": torch.cat(features).numpy().astype(np.float64),
        "residual": torch.cat(residuals).numpy().astype(np.float64),
        "phase": torch.cat(phases).numpy().astype(np.int64),
    }


def standardize_features(train: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = train.reshape(train.shape[0], -1)
    val_flat = val.reshape(val.shape[0], -1)
    mean = flat.mean(axis=0, keepdims=True)
    scale = flat.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    return (flat - mean) / scale, (val_flat - mean) / scale


def condition_ids(
    kind: str,
    phase: np.ndarray,
    horizon_bins: int,
    phase_bins: int,
) -> np.ndarray:
    samples, horizon = phase.shape
    horizon_id = np.floor(np.arange(horizon) * horizon_bins / horizon).astype(int)
    horizon_grid = np.broadcast_to(horizon_id[None], (samples, horizon))
    if kind == "D1":
        return np.zeros_like(phase)
    if kind == "D2":
        return horizon_grid
    if kind == "D3":
        return phase
    if kind in {"D4", "D4_perm"}:
        return horizon_grid * phase_bins + phase
    raise ValueError(kind)


def source_columns(target: int, n_vars: int, summaries: int = 4) -> np.ndarray:
    return np.concatenate(
        [
            np.arange(source * summaries, (source + 1) * summaries)
            for source in range(n_vars)
            if source != target
        ]
    )


def ridge_coefficients(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alpha: float,
) -> np.ndarray:
    n_groups = int(groups.max()) + 1
    coefficients = np.zeros((y.shape[2], n_groups, x.shape[1] - 4), dtype=np.float64)
    for target in range(y.shape[2]):
        columns = source_columns(target, y.shape[2])
        design = x[:, columns]
        for group in range(n_groups):
            xtx = np.zeros((design.shape[1], design.shape[1]), dtype=np.float64)
            xty = np.zeros(design.shape[1], dtype=np.float64)
            count = 0
            for horizon in range(y.shape[1]):
                mask = groups[:, horizon] == group
                if not np.any(mask):
                    continue
                selected = design[mask]
                response = y[mask, horizon, target]
                xtx += selected.T @ selected
                xty += selected.T @ response
                count += int(mask.sum())
            if count:
                normalized_xtx = xtx / count
                normalized_xty = xty / count
                coefficients[target, group] = np.linalg.solve(
                    normalized_xtx + alpha * np.eye(design.shape[1]),
                    normalized_xty,
                )
    return coefficients


def correction_from_coefficients(
    x: np.ndarray,
    groups: np.ndarray,
    coefficients: np.ndarray,
    n_vars: int,
) -> np.ndarray:
    correction = np.zeros((x.shape[0], groups.shape[1], n_vars), dtype=np.float64)
    for target in range(n_vars):
        design = x[:, source_columns(target, n_vars)]
        for horizon in range(groups.shape[1]):
            group = groups[:, horizon]
            correction[:, horizon, target] = np.einsum(
                "sf,sf->s",
                design,
                coefficients[target, group],
            )
    return correction


def select_alpha(
    x: np.ndarray,
    residual: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, list[dict[str, float]]]:
    boundary = int(0.8 * x.shape[0])
    scores = []
    for alpha in RIDGE_GRID:
        coefficients = ridge_coefficients(
            x[:boundary], residual[:boundary], groups[:boundary], alpha
        )
        correction = correction_from_coefficients(
            x[boundary:], groups[boundary:], coefficients, residual.shape[2]
        )
        error = residual[boundary:] - correction
        scores.append({"alpha": alpha, "mse": float(np.mean(error * error))})
    selected = min(scores, key=lambda row: (row["mse"], row["alpha"]))["alpha"]
    return float(selected), scores


def evaluate_model(
    kind: str,
    x_train: np.ndarray,
    train: dict[str, np.ndarray],
    x_val: np.ndarray,
    val: dict[str, np.ndarray],
    horizon_bins: int,
    phase_bins: int,
    seed: int,
) -> tuple[dict, np.ndarray, np.ndarray | None]:
    if kind == "D0":
        error = val["residual"]
        return {
            "model": kind,
            "description": MODEL_IDS[kind],
            "selected_alpha": None,
            "mse": float(np.mean(error * error)),
            "mae": float(np.mean(np.abs(error))),
        }, error, None

    train_phase = train["phase"].copy()
    if kind == "D4_perm":
        permutation = np.random.default_rng(seed).permutation(train_phase.shape[0])
        train_phase = train_phase[permutation]
    train_groups = condition_ids(kind, train_phase, horizon_bins, phase_bins)
    val_groups = condition_ids(kind, val["phase"], horizon_bins, phase_bins)
    alpha, alpha_scores = select_alpha(x_train, train["residual"], train_groups)
    coefficients = ridge_coefficients(
        x_train, train["residual"], train_groups, alpha
    )
    correction = correction_from_coefficients(
        x_val, val_groups, coefficients, val["residual"].shape[2]
    )
    error = val["residual"] - correction
    return {
        "model": kind,
        "description": MODEL_IDS[kind],
        "selected_alpha": alpha,
        "alpha_scores": alpha_scores,
        "mse": float(np.mean(error * error)),
        "mae": float(np.mean(np.abs(error))),
    }, error, coefficients


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
    n = len(reference)
    draws = []
    for _ in range(repetitions):
        indices = []
        while len(indices) < n:
            start = int(rng.integers(0, max(1, n - block + 1)))
            indices.extend(range(start, min(start + block, n)))
        selected = np.asarray(indices[:n])
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


def coefficient_variation(coefficients: np.ndarray | None) -> dict[str, float] | None:
    if coefficients is None or coefficients.shape[1] < 2:
        return None
    flat = coefficients.reshape(-1, coefficients.shape[-1])
    norms = np.linalg.norm(flat, axis=1)
    valid = norms > 1e-10
    flat = flat[valid] / norms[valid, None]
    if len(flat) < 2:
        return None
    cosine = flat @ flat.T
    upper = cosine[np.triu_indices(len(flat), 1)]
    signs = np.sign(flat)
    sign_agreement = np.mean(signs[:, None] == signs[None, :], axis=-1)
    sign_upper = sign_agreement[np.triu_indices(len(flat), 1)]
    return {
        "mean_pairwise_cosine": float(np.mean(upper)),
        "mean_pairwise_sign_agreement": float(np.mean(sign_upper)),
    }


def diagnose_dataset(args: argparse.Namespace, dataset: str) -> dict:
    model, model_args, checkpoint, record = load_frozen_model(dataset)
    if model_args.pred_len != args.pred_len:
        raise ValueError(
            f"Frozen checkpoint pred_len={model_args.pred_len}, requested {args.pred_len}"
        )
    print(f"[{dataset}] extracting frozen train residuals", flush=True)
    train = extract_split(model, model_args, "train", args.phase_bins)
    print(f"[{dataset}] extracting frozen validation residuals", flush=True)
    val = extract_split(model, model_args, "val", args.phase_bins)
    x_train, x_val = standardize_features(train["features"], val["features"])

    results, errors, coefficients = {}, {}, {}
    for kind in MODEL_IDS:
        print(f"[{dataset}] fitting {kind}: {MODEL_IDS[kind]}", flush=True)
        row, error, coefficient = evaluate_model(
            kind,
            x_train,
            train,
            x_val,
            val,
            args.horizon_bins,
            args.phase_bins,
            args.seed,
        )
        results[kind] = row
        errors[kind] = error
        coefficients[kind] = coefficient

    checkpoint_mse = float(record["best_mse"])
    reproduction_relative_error = abs(
        results["D0"]["mse"] - checkpoint_mse
    ) / checkpoint_mse
    if reproduction_relative_error > 1e-5:
        raise RuntimeError(
            f"{dataset} D0 does not reproduce the frozen validation MSE: "
            f"diagnostic={results['D0']['mse']}, record={checkpoint_mse}, "
            f"relative_error={reproduction_relative_error}"
        )

    d1_mse = results["D1"]["mse"]
    for kind in ("D2", "D3", "D4", "D4_perm"):
        results[kind]["improvement_over_D1_pct"] = (
            100.0 * (d1_mse - results[kind]["mse"]) / d1_mse
        )
    d4_gain = results["D4"]["improvement_over_D1_pct"]
    perm_gain = results["D4_perm"]["improvement_over_D1_pct"]
    result = {
        "dataset": dataset,
        "scope": "frozen_checkpoint_train_fit_validation_evaluation_no_test",
        "checkpoint": str(checkpoint),
        "checkpoint_recorded_validation_mse": checkpoint_mse,
        "D0_reproduction_relative_error": reproduction_relative_error,
        "train_origins": int(x_train.shape[0]),
        "validation_origins": int(x_val.shape[0]),
        "pred_len": args.pred_len,
        "horizon_bins": args.horizon_bins,
        "phase_bins": args.phase_bins,
        "residual_units": "data_loader_training_standardized_units",
        "source_summaries": [
            "last", "local4_mean", "period24_mean", "last_minus_period_lag"
        ],
        "models": results,
        "D4_bootstrap_vs_D1": moving_block_ci(
            errors["D1"], errors["D4"], args.bootstrap_block,
            args.bootstrap_repetitions, args.seed,
        ),
        "D4_phase_permutation_gain_retained_fraction": (
            None if abs(d4_gain) < 1e-12 else float(perm_gain / d4_gain)
        ),
        "coefficient_variation": {
            kind: coefficient_variation(coefficients[kind])
            for kind in ("D1", "D2", "D3", "D4")
        },
    }
    return result


def gate(results: list[dict]) -> dict:
    improvements = [
        row["models"]["D4"]["improvement_over_D1_pct"] for row in results
    ]
    better_main_effects = [
        row["models"]["D4"]["mse"] < min(
            row["models"]["D2"]["mse"], row["models"]["D3"]["mse"]
        )
        for row in results
    ]
    permutation_removed_half = []
    for row in results:
        d4 = row["models"]["D4"]["improvement_over_D1_pct"]
        perm = row["models"]["D4_perm"]["improvement_over_D1_pct"]
        permutation_removed_half.append(d4 > 0 and perm <= 0.5 * d4)
    return {
        "datasets_at_least_1pct": sum(value >= 1.0 for value in improvements),
        "macro_D4_over_D1_improvement_pct": float(np.mean(improvements)),
        "worst_D4_over_D1_improvement_pct": float(np.min(improvements)),
        "D4_better_than_D2_and_D3_all": all(better_main_effects),
        "permutation_removes_half_all": all(permutation_removed_half),
        "non_ett_confirmed": any(not row["dataset"].startswith("ETT") for row in results),
        "implementation_gate_passed": (
            sum(value >= 1.0 for value in improvements) >= 2
            and np.mean(improvements) >= 1.0
            and np.min(improvements) >= -0.5
            and all(
                row["D4_bootstrap_vs_D1"]["improvement_low_pct"] > 0
                for row in results
            )
            and all(better_main_effects)
            and all(permutation_removed_half)
            and any(not row["dataset"].startswith("ETT") for row in results)
        ),
        "note": "Non-ETT confirmation is mandatory before model implementation.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("ETTh1", "ETTh2"), default=("ETTh1", "ETTh2"))
    parser.add_argument("--pred-len", type=int, default=192)
    parser.add_argument("--horizon-bins", type=int, default=4)
    parser.add_argument("--phase-bins", type=int, default=6)
    parser.add_argument("--bootstrap-block", type=int, default=24)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if min(args.horizon_bins, args.phase_bins, args.bootstrap_block, args.bootstrap_repetitions) < 1:
        parser.error("bin counts, bootstrap block, and repetitions must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for dataset in args.datasets:
        result = diagnose_dataset(args, dataset)
        results.append(result)
        (args.output_dir / f"{dataset}_p{args.pred_len}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    payload = {
        "experiment": "GraphMamba_HPMRG_D0_D4_diagnostic_v1",
        "configuration": vars(args) | {"output_dir": str(args.output_dir)},
        "datasets": results,
        "gate": gate(results),
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["gate"], indent=2), flush=True)
    print(f"Saved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
