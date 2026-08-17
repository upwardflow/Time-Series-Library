#!/usr/bin/env python3
"""Frozen branch-normalized audit of GraphMamba selective dynamics.

For accepted periodic GraphMamba checkpoints, this script removes only the
within-sequence time variation of realized delta, B, or C on one patch branch.
It uses ordered validation data and never constructs the test split.  These
counterfactuals are diagnostic probes, not candidate model implementations.
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


OUTPUT_ROOT = ROOT / "logs" / "graphmamba_selective_dynamics"
FAMILIES = ("delta", "B", "C")
MODES = {"E0": None}
for _branch, _prefix in (("local", "L"), ("period", "P")):
    for _family, _suffix in (("delta", "D"), ("B", "B"), ("C", "C")):
        MODES[f"{_prefix}{_suffix}0"] = (_branch, _family)


def ordered_loader(args: object) -> DataLoader:
    dataset, _ = data_provider(args, "val")
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )


class SelectiveAuditController:
    def __init__(self) -> None:
        self.intervention: tuple[str, str] | None = None
        self.branch_queue: list[str] = []
        self.branch: str | None = None
        self.encoder_outputs: dict[str, torch.Tensor] = {}

    def start(self, mode: str) -> None:
        self.intervention = MODES[mode]
        self.branch_queue = ["local", "period"]
        self.branch = None
        self.encoder_outputs = {}

    def enter_encoder(self) -> str:
        if not self.branch_queue:
            raise RuntimeError("Unexpected extra encoder call in selective audit")
        self.branch = self.branch_queue.pop(0)
        return self.branch

    def active_family(self) -> str | None:
        if self.intervention is None or self.branch is None:
            return None
        branch, family = self.intervention
        return family if branch == self.branch else None

    def finish(self) -> None:
        if self.branch_queue:
            raise RuntimeError(f"Missing encoder calls for {self.branch_queue}")
        if set(self.encoder_outputs) != {"local", "period"}:
            raise RuntimeError(
                f"Incomplete encoder captures: {sorted(self.encoder_outputs)}"
            )


def explicit_mamba_forward(module, hidden_states, inference_params=None):
    if inference_params is not None:
        raise NotImplementedError("Selective audit does not use cached inference")
    controller: SelectiveAuditController = module._selective_audit_controller
    if controller.branch is None:
        raise RuntimeError("Mamba called outside a registered patch branch")

    _, seqlen, _ = hidden_states.shape
    xz = F.linear(hidden_states, module.in_proj.weight, module.in_proj.bias)
    x, z = xz.transpose(1, 2).chunk(2, dim=1)
    x = F.silu(
        F.conv1d(
            x,
            module.conv1d.weight,
            module.conv1d.bias,
            padding=module.d_conv - 1,
            groups=module.d_inner,
        )[..., :seqlen]
    )

    x_dbl = module.x_proj(rearrange(x, "b d l -> (b l) d"))
    dt_lowrank, B, C = torch.split(
        x_dbl, [module.dt_rank, module.d_state, module.d_state], dim=-1
    )
    dt_logits = module.dt_proj.weight @ dt_lowrank.t()
    dt_logits = rearrange(dt_logits, "d (b l) -> b d l", l=seqlen)
    delta = F.softplus(dt_logits + module.dt_proj.bias.float()[None, :, None])
    B = rearrange(B, "(b l) n -> b n l", l=seqlen).contiguous()
    C = rearrange(C, "(b l) n -> b n l", l=seqlen).contiguous()

    family = controller.active_family()
    if family == "delta":
        delta = delta.mean(dim=-1, keepdim=True).expand_as(delta).contiguous()
    elif family == "B":
        B = B.mean(dim=-1, keepdim=True).expand_as(B).contiguous()
    elif family == "C":
        C = C.mean(dim=-1, keepdim=True).expand_as(C).contiguous()

    y = selective_scan_fn(
        x,
        delta,
        -torch.exp(module.A_log.float()),
        B,
        C,
        module.D.float(),
        z=z,
        delta_bias=None,
        delta_softplus=False,
    )
    return module.out_proj(rearrange(y, "b d l -> b l d"))


def install_explicit_audit(model) -> SelectiveAuditController:
    controller = SelectiveAuditController()
    original_encoder_forward = model.encoder.forward

    def audited_encoder_forward(encoder, x, scan_mode=None):
        branch = controller.enter_encoder()
        output = original_encoder_forward(x, scan_mode=scan_mode)
        controller.encoder_outputs[branch] = output.detach()
        return output

    model.encoder.forward = MethodType(audited_encoder_forward, model.encoder)
    count = 0
    for module in model.modules():
        if module.__class__.__module__.startswith("mamba_ssm") and hasattr(module, "x_proj"):
            if not all(hasattr(module, name) for name in ("dt_proj", "A_log", "D", "conv1d")):
                continue
            module._selective_audit_controller = controller
            module.forward = MethodType(explicit_mamba_forward, module)
            count += 1
    if count == 0:
        raise RuntimeError("No compatible Mamba-1 modules found")
    return controller


def forecast(model, controller, mode, x, x_mark, batch_y, y_mark, pred_len, label_len):
    target = batch_y[:, -pred_len:].float().cuda()
    decoder = torch.cat(
        (batch_y[:, :label_len].float().cuda(), torch.zeros_like(target)), dim=1
    )
    controller.start(mode)
    prediction = model(x, x_mark, decoder, y_mark)
    controller.finish()
    captures = dict(controller.encoder_outputs)
    return prediction, target, captures


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
        "block_origins": int(block),
        "repetitions": int(repetitions),
    }


def origin_relative_rms(active: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    dimensions = tuple(range(1, baseline.ndim))
    numerator = (active.float() - baseline.float()).square().mean(dim=dimensions).sqrt()
    denominator = baseline.float().square().mean(dim=dimensions).sqrt().clamp_min(1e-8)
    return numerator / denominator


def diagnose_dataset(cli_args: argparse.Namespace, dataset: str) -> dict:
    model, model_args, checkpoint, record = load_frozen_model(dataset)
    if model_args.mamba_version != 1:
        raise ValueError("Selective dynamics audit requires Mamba-1")
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
    internal = {mode: [] for mode in MODES if mode != "E0"}
    targets = []
    explicit_first = None

    with torch.no_grad():
        for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            x = batch_x.float().cuda()
            x_mark = batch_x_mark.float().cuda()
            y_mark = batch_y_mark.float().cuda()
            baseline, target, base_states = forecast(
                model, controller, "E0", x, x_mark, batch_y, y_mark,
                model_args.pred_len, model_args.label_len,
            )
            if batch_index == 0:
                explicit_first = baseline.detach()
            predictions["E0"].append(baseline.cpu())
            targets.append(target.cpu())

            for mode, intervention in MODES.items():
                if mode == "E0":
                    continue
                active, _, active_states = forecast(
                    model, controller, mode, x, x_mark, batch_y, y_mark,
                    model_args.pred_len, model_args.label_len,
                )
                predictions[mode].append(active.cpu())
                branch, _ = intervention
                internal[mode].append(
                    origin_relative_rms(active_states[branch], base_states[branch]).cpu()
                )

    target = torch.cat(targets).numpy().astype(np.float64)
    pred = {
        mode: torch.cat(chunks).numpy().astype(np.float64)
        for mode, chunks in predictions.items()
    }
    internal_np = {
        mode: torch.cat(chunks).numpy().astype(np.float64)
        for mode, chunks in internal.items()
    }
    baseline_error = target - pred["E0"]
    baseline_mse = float(np.mean(baseline_error * baseline_error))
    baseline_origin_mse = np.mean(baseline_error * baseline_error, axis=(1, 2))
    forecast_rms = float(np.sqrt(np.mean(pred["E0"] * pred["E0"])))

    models = {}
    for mode in MODES:
        error = target - pred[mode]
        perturbation = pred[mode] - pred["E0"]
        origin_forecast_rms = np.sqrt(np.mean(perturbation * perturbation, axis=(1, 2)))
        mse = float(np.mean(error * error))
        models[mode] = {
            "mse": mse,
            "mae": float(np.mean(np.abs(error))),
            "mse_change_vs_E0_pct": 100.0 * (mse - baseline_mse) / baseline_mse,
            "forecast_perturbation_relative_rms": float(
                np.sqrt(np.mean(perturbation * perturbation)) / max(forecast_rms, 1e-12)
            ),
        }
        if mode != "E0":
            models[mode].update(
                {
                    "encoder_relative_rms_mean": float(internal_np[mode].mean()),
                    "encoder_relative_rms_std": float(internal_np[mode].std()),
                    "internal_response_error_pearson": correlation(
                        internal_np[mode], baseline_origin_mse
                    ),
                    "internal_response_error_spearman": rank_correlation(
                        internal_np[mode], baseline_origin_mse
                    ),
                    "forecast_response_error_pearson": correlation(
                        origin_forecast_rms, baseline_origin_mse
                    ),
                }
            )

    family_results = {}
    family_modes = {
        "delta": ("LD0", "PD0"),
        "B": ("LB0", "PB0"),
        "C": ("LC0", "PC0"),
    }
    for family, (local_mode, period_mode) in family_modes.items():
        local = internal_np[local_mode]
        period = internal_np[period_mode]
        local_mean = float(local.mean())
        period_mean = float(period.mean())
        relative_difference = abs(local_mean - period_mean) / max(
            local_mean, period_mean, 1e-12
        )
        ci = moving_block_difference_ci(
            local,
            period,
            cli_args.bootstrap_block,
            cli_args.bootstrap_repetitions,
            cli_args.seed,
        )
        material = max(local_mean, period_mean) >= cli_args.min_internal_response
        distinguished = (
            relative_difference >= cli_args.min_relative_difference
            and (ci["low"] > 0 or ci["high"] < 0)
        )
        sign = 1 if local_mean > period_mean else (-1 if local_mean < period_mean else 0)
        family_results[family] = {
            "local_encoder_relative_rms_mean": local_mean,
            "period_encoder_relative_rms_mean": period_mean,
            "relative_difference": float(relative_difference),
            "local_minus_period_sign": sign,
            "paired_block_bootstrap": ci,
            "material": bool(material),
            "distinguished": bool(distinguished),
            "dataset_passed": bool(material and distinguished),
        }

    reproduction_relative_error = abs(baseline_mse - float(record["best_mse"])) / float(
        record["best_mse"]
    )
    if reproduction_relative_error > 1e-5:
        raise RuntimeError(
            f"{dataset} explicit E0 MSE mismatch: {baseline_mse} vs "
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
        "local_patch_count": int(model.local_patch_count),
        "period_patch_count": int(model.period_patch_count),
        "explicit_fused_first_batch_max_abs": first_max_abs,
        "E0_checkpoint_mse_reproduction_relative_error": reproduction_relative_error,
        "models": models,
        "selective_families": family_results,
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
    parser.add_argument("--min-internal-response", type=float, default=0.02)
    parser.add_argument("--min-relative-difference", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in args.datasets:
        print(f"[{dataset}] running frozen selective-dynamics audit", flush=True)
        row = diagnose_dataset(args, dataset)
        rows.append(row)
        (args.output_dir / f"{dataset}_p192.json").write_text(
            json.dumps(row, indent=2) + "\n", encoding="utf-8"
        )

    eligible = []
    for family in FAMILIES:
        family_rows = [row["selective_families"][family] for row in rows]
        if (
            len(family_rows) == 2
            and all(item["dataset_passed"] for item in family_rows)
            and family_rows[0]["local_minus_period_sign"]
            == family_rows[1]["local_minus_period_sign"]
            != 0
        ):
            eligible.append(family)
    payload = {
        "experiment": "GraphMamba_frozen_selective_dynamics_branch_audit_v0",
        "configuration": vars(args) | {"output_dir": str(args.output_dir)},
        "interventions": MODES,
        "datasets": rows,
        "gate": {
            "eligible_families_for_post_result_novelty_search": eligible,
            "proceed_to_candidate_design": False,
            "note": "Passing only authorizes a new operation-level novelty search.",
        },
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["gate"], indent=2), flush=True)
    print(f"Saved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
