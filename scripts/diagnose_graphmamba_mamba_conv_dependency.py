#!/usr/bin/env python3
"""Frozen validation-only audit of Mamba temporal-convolution dependency.

The accepted periodic GraphMamba checkpoint is evaluated through an explicit
Mamba-1 path.  Interventions remove only past-lag convolution taps on the local
and/or period patch branch while retaining the learned current tap, bias, SiLU,
selective scan, graph path, and prediction head.  The test split is never built.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_provider.data_factory import data_provider
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from scripts.diagnose_horizon_phase_relation_bound import load_frozen_model


OUTPUT_ROOT = ROOT / "logs" / "graphmamba_mamba_conv_dependency"
MODES = {
    "E0": {"local": 1.0, "period": 1.0},
    "EL0": {"local": 0.0, "period": 1.0},
    "EP0": {"local": 1.0, "period": 0.0},
    "EB0": {"local": 0.0, "period": 0.0},
}


def ordered_loader(args: object) -> DataLoader:
    dataset, _ = data_provider(args, "val")
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )


class ConvAuditController:
    def __init__(self) -> None:
        self.mode = MODES["E0"]
        self.branch_queue: list[str] = []
        self.branch: str | None = None
        self.capture = False
        self.records: dict[str, list[torch.Tensor]] = {"local": [], "period": []}

    def start(self, mode: str, capture: bool = False) -> None:
        self.mode = MODES[mode]
        self.branch_queue = ["local", "period"]
        self.branch = None
        self.capture = capture
        self.records = {"local": [], "period": []}

    def enter_encoder(self) -> None:
        if not self.branch_queue:
            raise RuntimeError("Unexpected extra encoder call in convolution audit")
        self.branch = self.branch_queue.pop(0)

    def finish(self) -> None:
        if self.branch_queue:
            raise RuntimeError(f"Missing encoder calls for {self.branch_queue}")


def explicit_mamba_forward(module, hidden_states, inference_params=None):
    if inference_params is not None:
        raise NotImplementedError("The frozen convolution audit does not use cached inference")
    controller: ConvAuditController = module._conv_audit_controller
    if controller.branch is None:
        raise RuntimeError("Mamba called outside a registered patch branch")
    batch, seqlen, _ = hidden_states.shape
    xz = F.linear(hidden_states, module.in_proj.weight, module.in_proj.bias)
    x, z = xz.transpose(1, 2).chunk(2, dim=1)

    weight = module.conv1d.weight
    full = F.silu(
        F.conv1d(
            x,
            weight,
            module.conv1d.bias,
            padding=module.d_conv - 1,
            groups=module.d_inner,
        )[..., :seqlen]
    )
    current_weight = torch.zeros_like(weight)
    current_weight[..., -1:] = weight[..., -1:]
    current = F.silu(
        F.conv1d(
            x,
            current_weight,
            module.conv1d.bias,
            padding=module.d_conv - 1,
            groups=module.d_inner,
        )[..., :seqlen]
    )
    temporal = full - current
    scale = controller.mode[controller.branch]
    active = current + scale * temporal

    if controller.capture:
        numerator = temporal.float().square().mean(dim=(1, 2)).sqrt()
        denominator = full.float().square().mean(dim=(1, 2)).sqrt().clamp_min(1e-8)
        controller.records[controller.branch].append((numerator / denominator).detach())

    x_dbl = module.x_proj(rearrange(active, "b d l -> (b l) d"))
    dt, B, C = torch.split(
        x_dbl, [module.dt_rank, module.d_state, module.d_state], dim=-1
    )
    dt = module.dt_proj.weight @ dt.t()
    dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
    B = rearrange(B, "(b l) n -> b n l", l=seqlen).contiguous()
    C = rearrange(C, "(b l) n -> b n l", l=seqlen).contiguous()
    y = selective_scan_fn(
        active,
        dt,
        -torch.exp(module.A_log.float()),
        B,
        C,
        module.D.float(),
        z=z,
        delta_bias=module.dt_proj.bias.float(),
        delta_softplus=True,
    )
    return module.out_proj(rearrange(y, "b d l -> b l d"))


def install_explicit_audit(model) -> ConvAuditController:
    controller = ConvAuditController()
    original_encoder_forward = model.encoder.forward

    def audited_encoder_forward(encoder, x, scan_mode=None):
        controller.enter_encoder()
        return original_encoder_forward(x, scan_mode=scan_mode)

    model.encoder.forward = MethodType(audited_encoder_forward, model.encoder)
    count = 0
    for module in model.modules():
        if module.__class__.__module__.startswith("mamba_ssm") and hasattr(module, "x_proj"):
            if not all(hasattr(module, name) for name in ("dt_proj", "A_log", "D", "conv1d")):
                continue
            module._conv_audit_controller = controller
            module.forward = MethodType(explicit_mamba_forward, module)
            count += 1
    if count == 0:
        raise RuntimeError("No compatible Mamba-1 modules found")
    return controller


def forecast(model, controller, mode, x, x_mark, batch_y, y_mark, pred_len, label_len, capture=False):
    y = batch_y[:, -pred_len:].float().cuda()
    decoder = torch.cat(
        (batch_y[:, :label_len].float().cuda(), torch.zeros_like(y)), dim=1
    )
    controller.start(mode, capture=capture)
    prediction = model(x, x_mark, decoder, y_mark)
    controller.finish()
    ratios = None
    if capture:
        ratios = {
            branch: torch.stack(values).mean(dim=0).cpu()
            for branch, values in controller.records.items()
        }
    return prediction, y, ratios


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return correlation(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))


def moving_block_difference_ci(
    local: np.ndarray,
    period: np.ndarray,
    block: int,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    difference = local - period
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        indices = []
        while len(indices) < len(difference):
            start = int(rng.integers(0, max(1, len(difference) - block + 1)))
            indices.extend(range(start, min(start + block, len(difference))))
        draws.append(float(difference[np.asarray(indices[: len(difference)])].mean()))
    low, high = np.percentile(draws, (2.5, 97.5))
    return {
        "mean_local_minus_period": float(difference.mean()),
        "low": float(low),
        "high": float(high),
        "block_origins": block,
        "repetitions": repetitions,
    }


def diagnose_dataset(cli_args: argparse.Namespace, dataset: str) -> dict:
    model, model_args, checkpoint, record = load_frozen_model(dataset)
    if model_args.mamba_version != 1 or model_args.d_conv < 2:
        raise ValueError("Convolution dependency audit requires Mamba-1 with d_conv >= 2")
    loader = ordered_loader(model_args)
    first = next(iter(loader))
    first_x = first[0].float().cuda()
    with torch.no_grad():
        fused_prediction = model(
            first_x,
            first[2].float().cuda(),
            torch.cat(
                (
                    first[1][:, : model_args.label_len].float().cuda(),
                    torch.zeros_like(first[1][:, -model_args.pred_len :].float().cuda()),
                ),
                dim=1,
            ),
            first[3].float().cuda(),
        )

    controller = install_explicit_audit(model)
    predictions = {mode: [] for mode in MODES}
    targets, local_ratios, period_ratios = [], [], []
    explicit_first = None
    with torch.no_grad():
        for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            x = batch_x.float().cuda()
            x_mark = batch_x_mark.float().cuda()
            y_mark = batch_y_mark.float().cuda()
            baseline, y, ratios = forecast(
                model, controller, "E0", x, x_mark, batch_y, y_mark,
                model_args.pred_len, model_args.label_len, capture=True,
            )
            if batch_index == 0:
                explicit_first = baseline.detach()
            predictions["E0"].append(baseline.cpu())
            targets.append(y.cpu())
            local_ratios.append(ratios["local"])
            period_ratios.append(ratios["period"])
            for mode in ("EL0", "EP0", "EB0"):
                output, _, _ = forecast(
                    model, controller, mode, x, x_mark, batch_y, y_mark,
                    model_args.pred_len, model_args.label_len,
                )
                predictions[mode].append(output.cpu())

    target = torch.cat(targets).numpy().astype(np.float64)
    pred = {
        mode: torch.cat(chunks).numpy().astype(np.float64)
        for mode, chunks in predictions.items()
    }
    baseline_error = target - pred["E0"]
    baseline_origin_mse = np.mean(baseline_error * baseline_error, axis=(1, 2))
    forecast_rms = float(np.sqrt(np.mean(pred["E0"] * pred["E0"])))
    results = {}
    perturbations = {}
    for mode in MODES:
        error = target - pred[mode]
        perturbation = pred[mode] - pred["E0"]
        origin_rms = np.sqrt(np.mean(perturbation * perturbation, axis=(1, 2)))
        perturbations[mode] = origin_rms
        mse = float(np.mean(error * error))
        results[mode] = {
            "mse": mse,
            "mae": float(np.mean(np.abs(error))),
            "mse_change_vs_E0_pct": 100.0 * (mse - np.mean(baseline_error ** 2)) / np.mean(baseline_error ** 2),
            "forecast_perturbation_rms": float(np.sqrt(np.mean(perturbation * perturbation))),
            "forecast_perturbation_relative_rms": float(
                np.sqrt(np.mean(perturbation * perturbation)) / max(forecast_rms, 1e-12)
            ),
            "perturbation_error_pearson": correlation(origin_rms, baseline_origin_mse),
            "perturbation_error_spearman": rank_correlation(origin_rms, baseline_origin_mse),
        }

    local = torch.cat(local_ratios).numpy().astype(np.float64)
    period = torch.cat(period_ratios).numpy().astype(np.float64)
    ci = moving_block_difference_ci(
        local, period, cli_args.bootstrap_block,
        cli_args.bootstrap_repetitions, cli_args.seed,
    )
    internal_relative_difference = abs(local.mean() - period.mean()) / max(
        local.mean(), period.mean(), 1e-12
    )
    material = (
        max(local.mean(), period.mean()) >= cli_args.min_internal_ratio
        and max(
            results["EL0"]["forecast_perturbation_relative_rms"],
            results["EP0"]["forecast_perturbation_relative_rms"],
        ) >= cli_args.min_forecast_perturbation
    )
    distinguished = (
        internal_relative_difference >= cli_args.min_relative_difference
        and (ci["low"] > 0 or ci["high"] < 0)
    )
    reproduction_relative_error = abs(
        results["E0"]["mse"] - float(record["best_mse"])
    ) / float(record["best_mse"])
    if reproduction_relative_error > 1e-5:
        raise RuntimeError(
            f"{dataset} explicit E0 MSE mismatch: {results['E0']['mse']} vs "
            f"{record['best_mse']} ({reproduction_relative_error})"
        )
    first_max_abs = float((explicit_first - fused_prediction).abs().max())
    if first_max_abs > 2e-5:
        raise RuntimeError(f"{dataset} explicit/fused max difference {first_max_abs}")
    return {
        "dataset": dataset,
        "scope": "ordered_validation_only_no_test_frozen_checkpoint",
        "checkpoint": str(checkpoint),
        "validation_origins": int(len(target)),
        "mamba_version": int(model_args.mamba_version),
        "d_conv": int(model_args.d_conv),
        "local_stride_hours": int(model_args.periodic_local_stride),
        "period_stride_hours": int(model_args.periodic_period_stride),
        "explicit_fused_first_batch_max_abs": first_max_abs,
        "E0_checkpoint_mse_reproduction_relative_error": reproduction_relative_error,
        "internal_temporal_component_relative_rms": {
            "local_mean": float(local.mean()),
            "period_mean": float(period.mean()),
            "relative_difference": float(internal_relative_difference),
            "paired_block_bootstrap": ci,
        },
        "models": results,
        "gate": {
            "material": bool(material),
            "distinguished": bool(distinguished),
            "dataset_passed": bool(material and distinguished),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=("ETTh1", "ETTh2"),
        default=("ETTh1", "ETTh2"),
    )
    parser.add_argument("--bootstrap-block", type=int, default=24)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--min-internal-ratio", type=float, default=0.05)
    parser.add_argument("--min-forecast-perturbation", type=float, default=0.002)
    parser.add_argument("--min-relative-difference", type=float, default=0.10)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in args.datasets:
        print(f"[{dataset}] running frozen convolution audit", flush=True)
        row = diagnose_dataset(args, dataset)
        rows.append(row)
        (args.output_dir / f"{dataset}_p192.json").write_text(
            json.dumps(row, indent=2) + "\n", encoding="utf-8"
        )
    passed = all(row["gate"]["dataset_passed"] for row in rows)
    payload = {
        "experiment": "GraphMamba_frozen_Mamba1_temporal_convolution_dependency_v0",
        "configuration": vars(args) | {"output_dir": str(args.output_dir)},
        "interventions": MODES,
        "datasets": rows,
        "gate": {
            "all_datasets_material_and_distinguished": bool(passed),
            "proceed_to_controlled_training": bool(passed),
        },
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["gate"], indent=2), flush=True)
    print(f"Saved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
