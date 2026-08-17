#!/usr/bin/env python3
"""Attribute GraphMamba's scale bottleneck without training or test access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_provider.data_factory import data_provider
from scripts.diagnose_horizon_phase_relation_bound import load_frozen_model


OUTPUT_ROOT = ROOT / "logs" / "graphmamba_scale_bottleneck"


def ordered_loader(args: object) -> DataLoader:
    dataset, _ = data_provider(args, "val")
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )


class Capture:
    def __init__(self, model) -> None:
        self.queue: list[str] = []
        self.encoder_outputs: dict[str, torch.Tensor] = {}
        self.head_input: torch.Tensor | None = None
        original = model.encoder.forward

        def encoder_forward(encoder, x, scan_mode=None):
            if not self.queue:
                raise RuntimeError("Unexpected encoder call")
            branch = self.queue.pop(0)
            output = original(x, scan_mode=scan_mode)
            self.encoder_outputs[branch] = output
            return output

        model.encoder.forward = MethodType(encoder_forward, model.encoder)

        def head_pre_hook(_module, inputs):
            self.head_input = inputs[0]

        model.head.register_forward_pre_hook(head_pre_hook)

    def start(self) -> None:
        self.queue = ["local", "period"]
        self.encoder_outputs = {}
        self.head_input = None

    def finish(self) -> None:
        if self.queue or set(self.encoder_outputs) != {"local", "period"}:
            raise RuntimeError("Incomplete scale capture")
        if self.head_input is None:
            raise RuntimeError("Head input was not captured")


def vector_stats(local_grads, period_grads) -> tuple[float, float]:
    dot = torch.zeros((), device="cuda", dtype=torch.float64)
    local_sq = torch.zeros_like(dot)
    period_sq = torch.zeros_like(dot)
    for local, period in zip(local_grads, period_grads):
        if local is None or period is None:
            continue
        local64 = local.detach().double()
        period64 = period.detach().double()
        dot += torch.sum(local64 * period64)
        local_sq += torch.sum(local64 * local64)
        period_sq += torch.sum(period64 * period64)
    local_norm = torch.sqrt(local_sq)
    period_norm = torch.sqrt(period_sq)
    cosine = dot / (local_norm * period_norm).clamp_min(1e-30)
    ratio = period_norm / local_norm.clamp_min(1e-30)
    return float(cosine.cpu()), float(ratio.cpu())


def parameter_group(name: str) -> str:
    if ".in_proj." in name:
        return "input_gate_projection"
    if ".conv1d." in name:
        return "temporal_convolution"
    if ".x_proj." in name or ".dt_proj." in name:
        return "selective_projection"
    if name.endswith(".A_log"):
        return "state_generator_A"
    if name.endswith(".D"):
        return "skip_D"
    if ".out_proj." in name:
        return "output_projection"
    return "block_norm_ffn"


def bootstrap_mean(values: np.ndarray, repetitions: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(draws, (2.5, 97.5))
    return {"mean": float(values.mean()), "low": float(low), "high": float(high)}


def batch_forward(model, capture, args, batch):
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    x = batch_x.float().cuda()
    target = batch_y[:, -args.pred_len :].float().cuda()
    decoder = torch.cat(
        (
            batch_y[:, : args.label_len].float().cuda(),
            torch.zeros_like(target),
        ),
        dim=1,
    )
    capture.start()
    prediction = model(
        x,
        batch_x_mark.float().cuda(),
        decoder,
        batch_y_mark.float().cuda(),
    )
    capture.finish()
    return prediction, target


def diagnose_dataset(cli_args: argparse.Namespace, dataset: str) -> dict:
    model, args, checkpoint, record = load_frozen_model(dataset)
    loader = ordered_loader(args)
    capture = Capture(model)
    named_parameters = tuple(
        (name, parameter) for name, parameter in model.encoder.named_parameters()
        if parameter.requires_grad
    )
    parameters = tuple(parameter for _, parameter in named_parameters)
    selected = set(
        np.linspace(0, len(loader) - 1, min(cli_args.gradient_batches, len(loader)), dtype=int).tolist()
    )

    cosines, norm_ratios = [], []
    group_names = tuple(sorted({parameter_group(name) for name, _ in named_parameters}))
    group_cosines = {name: [] for name in group_names}
    group_norm_ratios = {name: [] for name in group_names}
    contribution_ratios = []
    baseline_errors, no_local_errors, no_period_errors = [], [], []

    for batch_index, batch in enumerate(loader):
        if batch_index not in selected:
            continue
        model.zero_grad(set_to_none=True)
        prediction, target = batch_forward(model, capture, args, batch)
        loss = F.mse_loss(prediction, target)
        local_state = capture.encoder_outputs["local"]
        period_state = capture.encoder_outputs["period"]
        local_signal, period_signal = torch.autograd.grad(
            loss, (local_state, period_state), retain_graph=True
        )
        local_grads = torch.autograd.grad(
            local_state,
            parameters,
            grad_outputs=local_signal,
            retain_graph=True,
            allow_unused=True,
        )
        period_grads = torch.autograd.grad(
            period_state,
            parameters,
            grad_outputs=period_signal,
            allow_unused=True,
        )
        cosine, norm_ratio = vector_stats(local_grads, period_grads)
        cosines.append(cosine)
        norm_ratios.append(norm_ratio)
        for group in group_names:
            indices = [
                index for index, (name, _) in enumerate(named_parameters)
                if parameter_group(name) == group
            ]
            group_cosine, group_ratio = vector_stats(
                [local_grads[index] for index in indices],
                [period_grads[index] for index in indices],
            )
            group_cosines[group].append(group_cosine)
            group_norm_ratios[group].append(group_ratio)

        with torch.no_grad():
            fused = capture.head_input
            flattened = fused.permute(0, 1, 3, 2).flatten(start_dim=-2)
            split = model.local_patch_count * model.d_model
            weight = model.head.linear.weight
            local_contribution = F.linear(flattened[..., :split], weight[:, :split])
            period_contribution = F.linear(flattened[..., split:], weight[:, split:])
            local_contribution = local_contribution.permute(0, 2, 1)
            period_contribution = period_contribution.permute(0, 2, 1)
            ratio = (
                period_contribution.float().square().mean(dim=(1, 2)).sqrt()
                / local_contribution.float().square().mean(dim=(1, 2)).sqrt().clamp_min(1e-8)
            )
            contribution_ratios.extend(ratio.cpu().tolist())

            scale = torch.sqrt(
                torch.var(
                    batch[0].float().cuda()
                    - batch[0].float().cuda().mean(dim=1, keepdim=True),
                    dim=1,
                    keepdim=True,
                    unbiased=False,
                )
                + 1e-5
            ).detach()
            local_physical = local_contribution * scale
            period_physical = period_contribution * scale
            baseline_errors.extend(
                ((prediction - target).square().mean(dim=(1, 2))).cpu().tolist()
            )
            no_local_errors.extend(
                ((prediction - local_physical - target).square().mean(dim=(1, 2))).cpu().tolist()
            )
            no_period_errors.extend(
                ((prediction - period_physical - target).square().mean(dim=(1, 2))).cpu().tolist()
            )

    cosines_np = np.asarray(cosines, dtype=np.float64)
    ratios_np = np.asarray(norm_ratios, dtype=np.float64)
    contribution_np = np.asarray(contribution_ratios, dtype=np.float64)
    baseline_np = np.asarray(baseline_errors, dtype=np.float64)
    no_local_np = np.asarray(no_local_errors, dtype=np.float64)
    no_period_np = np.asarray(no_period_errors, dtype=np.float64)

    split = model.local_patch_count * model.d_model
    head_weight = model.head.linear.weight.detach().float()
    local_weight_norm = float(head_weight[:, :split].square().sum().sqrt().cpu())
    period_weight_norm = float(head_weight[:, split:].square().sum().sqrt().cpu())
    conflict = (
        cosines_np.mean() < cli_args.conflict_cosine
        and bootstrap_mean(cosines_np, cli_args.bootstrap_repetitions, cli_args.seed)["high"] < 0
        and 0.10 <= ratios_np.mean() <= 10.0
    )
    underidentified = (
        not conflict
        and ratios_np.mean() < cli_args.max_period_gradient_ratio
        and contribution_np.mean() < cli_args.max_period_contribution_ratio
    )
    component_rows = {}
    for group in group_names:
        group_cosine_np = np.asarray(group_cosines[group], dtype=np.float64)
        group_ratio_np = np.asarray(group_norm_ratios[group], dtype=np.float64)
        cosine_ci = bootstrap_mean(
            group_cosine_np, cli_args.bootstrap_repetitions, cli_args.seed
        )
        component_rows[group] = {
            "local_period_cosine": cosine_ci,
            "period_over_local_norm_mean": float(group_ratio_np.mean()),
            "conflict": bool(
                cosine_ci["mean"] < cli_args.conflict_cosine
                and cosine_ci["high"] < 0
            ),
        }
    return {
        "dataset": dataset,
        "scope": "evenly_spaced_ordered_validation_only_no_test_frozen_checkpoint",
        "checkpoint": str(checkpoint),
        "selected_batches": len(selected),
        "selected_origins": len(contribution_np),
        "checkpoint_best_mse": float(record["best_mse"]),
        "gradient": {
            "local_period_cosine": bootstrap_mean(
                cosines_np, cli_args.bootstrap_repetitions, cli_args.seed
            ),
            "period_over_local_norm_mean": float(ratios_np.mean()),
            "period_over_local_norm_median": float(np.median(ratios_np)),
            "parameter_groups": component_rows,
        },
        "head": {
            "period_over_local_marginal_rms_mean": float(contribution_np.mean()),
            "period_over_local_marginal_rms_median": float(np.median(contribution_np)),
            "period_over_local_total_weight_norm": period_weight_norm / max(local_weight_norm, 1e-12),
            "period_over_local_per_token_weight_rms": (
                period_weight_norm / np.sqrt(model.period_patch_count)
            ) / max(local_weight_norm / np.sqrt(model.local_patch_count), 1e-12),
        },
        "frozen_branch_removal": {
            "baseline_mse": float(baseline_np.mean()),
            "no_local_mse": float(no_local_np.mean()),
            "no_period_mse": float(no_period_np.mean()),
            "no_local_change_pct": float(100 * (no_local_np.mean() - baseline_np.mean()) / baseline_np.mean()),
            "no_period_change_pct": float(100 * (no_period_np.mean() - baseline_np.mean()) / baseline_np.mean()),
        },
        "attribution": {
            "shared_core_conflict": bool(conflict),
            "period_role_underidentified": bool(underidentified),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("ETTh1", "ETTh2"), default=("ETTh1", "ETTh2"))
    parser.add_argument("--gradient-batches", type=int, default=32)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--conflict-cosine", type=float, default=-0.10)
    parser.add_argument("--max-period-gradient-ratio", type=float, default=0.25)
    parser.add_argument("--max-period-contribution-ratio", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in args.datasets:
        print(f"[{dataset}] attributing scale bottleneck", flush=True)
        row = diagnose_dataset(args, dataset)
        rows.append(row)
        (args.output_dir / f"{dataset}_p192.json").write_text(
            json.dumps(row, indent=2) + "\n", encoding="utf-8"
        )
    all_conflict = all(row["attribution"]["shared_core_conflict"] for row in rows)
    all_underidentified = all(
        row["attribution"]["period_role_underidentified"] for row in rows
    )
    generator_shared = all(
        not row["gradient"]["parameter_groups"]["state_generator_A"]["conflict"]
        for row in rows
    )
    interface_groups = (
        "input_gate_projection", "temporal_convolution", "selective_projection",
        "output_projection",
    )
    common_conflict_interfaces = [
        group for group in interface_groups
        if all(row["gradient"]["parameter_groups"][group]["conflict"] for row in rows)
    ]
    payload = {
        "experiment": "GraphMamba_scale_bottleneck_attribution_v0",
        "configuration": vars(args) | {"output_dir": str(args.output_dir)},
        "datasets": rows,
        "decision": {
            "shared_core_conflict": bool(all_conflict),
            "period_role_underidentified": bool(all_underidentified),
            "selected_locus": (
                "shared_core" if all_conflict else "scale_responsibility_interface" if all_underidentified else "none_mixed"
            ),
            "shared_generator_scale_interface_candidate_admissible": bool(
                all_conflict and generator_shared and common_conflict_interfaces
            ),
            "common_conflict_interfaces": common_conflict_interfaces,
        },
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2), flush=True)
    print(f"Saved: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
