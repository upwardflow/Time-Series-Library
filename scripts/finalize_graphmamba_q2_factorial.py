#!/usr/bin/env python3
"""Aggregate and audit the Q2 periodic-backbone x TimeRole factorial."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "logs" / "graphmamba_q2_factorial" / "validation"
OUTPUT = ROOT / "logs" / "graphmamba_q2_factorial"
REPORT = ROOT / "experiment_results" / "GraphMamba_Q2_factorial_validation.md"
DATASETS = ("ETTh1", "ETTh2")
HORIZONS = (192, 720)
SEEDS = (2021, 2022, 2023)
VARIANTS = ("b", "p", "c", "pc")


def name(dataset: str, horizon: int, variant: str, seed: int) -> str:
    return f"q2f_{dataset.lower()}_p{horizon}_{variant}_s{seed}"


def load_records() -> dict[tuple[str, int, int], dict[str, dict[str, object]]]:
    grouped: dict[tuple[str, int, int], dict[str, dict[str, object]]] = defaultdict(dict)
    failures: list[str] = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            for seed in SEEDS:
                for variant in VARIANTS:
                    candidate = name(dataset, horizon, variant, seed)
                    path = SOURCE / f"{candidate}.json"
                    if not path.is_file():
                        failures.append(f"missing {candidate}")
                        continue
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if payload.get("status") != "completed":
                        failures.append(f"incomplete {candidate}")
                        continue
                    if payload.get("final_test") is not False:
                        failures.append(f"test-access violation {candidate}")
                    grouped[(dataset, horizon, seed)][variant] = payload
    if failures:
        raise RuntimeError("; ".join(failures))
    return grouped


def improvement(control: float, candidate: float) -> float:
    return 100.0 * (control - candidate) / control


def main() -> int:
    grouped = load_records()
    rows: list[dict[str, object]] = []
    for (dataset, horizon, seed), records in sorted(grouped.items()):
        mse = {variant: float(records[variant]["best_mse"]) for variant in VARIANTS}
        mae = {variant: float(records[variant]["best_mae"]) for variant in VARIANTS}
        rows.append(
            {
                "dataset": dataset,
                "horizon": horizon,
                "seed": seed,
                **{f"{variant}_mse": mse[variant] for variant in VARIANTS},
                **{f"{variant}_mae": mae[variant] for variant in VARIANTS},
                "periodic_without_memory_mse_pct": improvement(mse["b"], mse["p"]),
                "periodic_with_memory_mse_pct": improvement(mse["c"], mse["pc"]),
                "memory_without_periodic_mse_pct": improvement(mse["b"], mse["c"]),
                "memory_with_periodic_mse_pct": improvement(mse["p"], mse["pc"]),
                "factorial_interaction_mse": mse["p"] + mse["c"] - mse["b"] - mse["pc"],
                "full_vs_best_single_mse_pct": improvement(min(mse["p"], mse["c"]), mse["pc"]),
                "periodic_without_memory_mae_pct": improvement(mae["b"], mae["p"]),
                "periodic_with_memory_mae_pct": improvement(mae["c"], mae["pc"]),
                "memory_without_periodic_mae_pct": improvement(mae["b"], mae["c"]),
                "memory_with_periodic_mae_pct": improvement(mae["p"], mae["pc"]),
                "factorial_interaction_mae": mae["p"] + mae["c"] - mae["b"] - mae["pc"],
                "full_vs_best_single_mae_pct": improvement(min(mae["p"], mae["c"]), mae["pc"]),
            }
        )

    fields = list(rows[0])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "factorial_pairs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metric_fields = (
        "periodic_without_memory_mse_pct",
        "periodic_with_memory_mse_pct",
        "memory_without_periodic_mse_pct",
        "memory_with_periodic_mse_pct",
        "factorial_interaction_mse",
        "full_vs_best_single_mse_pct",
        "periodic_without_memory_mae_pct",
        "periodic_with_memory_mae_pct",
        "memory_without_periodic_mae_pct",
        "memory_with_periodic_mae_pct",
        "factorial_interaction_mae",
        "full_vs_best_single_mae_pct",
    )
    summary = {
        metric: {
            "mean": sum(float(row[metric]) for row in rows) / len(rows),
            "positive_count": sum(float(row[metric]) > 0 for row in rows),
            "total": len(rows),
        }
        for metric in metric_fields
    }
    summary["preregistered_readout"] = {
        "memory_compatible_with_periodic": (
            summary["memory_with_periodic_mse_pct"]["positive_count"] >= 10
            and summary["memory_with_periodic_mse_pct"]["mean"] >= 1.0
        ),
        "periodic_adds_with_memory": (
            summary["periodic_with_memory_mse_pct"]["positive_count"] >= 7
            and summary["periodic_with_memory_mse_pct"]["mean"] > 0.0
        ),
        "interaction_not_materially_negative": (
            summary["factorial_interaction_mse"]["mean"] >= -0.005
        ),
        "memory_mae_not_systematically_worse_with_periodic": (
            summary["memory_with_periodic_mae_pct"]["positive_count"] >= 7
            and summary["memory_with_periodic_mae_pct"]["mean"] >= 0.0
        ),
    }
    (OUTPUT / "factorial_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# GraphMamba Q2 periodic × TimeRole factorial validation",
        "",
        "## Protocol",
        "",
        "- Validation only; no test evaluation.",
        "- ETTh1/ETTh2 × horizons 192/720 × seeds 2021/2022/2023.",
        "- All variants load 336 points; all backbones process only the recent 96 points; only TimeRole reads the old 240 points.",
        "- `b`: independent dual patches; `p`: periodic multi-resolution backbone; `c`: `b` + TimeRole; `pc`: `p` + TimeRole.",
        "",
        "## Paired results",
        "",
        "| Dataset | H | Seed | B MSE | P MSE | C MSE | PC MSE | P→PC | C→PC | Interaction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['horizon']} | {row['seed']} | "
            f"{row['b_mse']:.6f} | {row['p_mse']:.6f} | {row['c_mse']:.6f} | "
            f"{row['pc_mse']:.6f} | {row['memory_with_periodic_mse_pct']:+.3f}% | "
            f"{row['periodic_with_memory_mse_pct']:+.3f}% | "
            f"{row['factorial_interaction_mse']:+.6f} |"
        )
    lines.extend(["", "## Aggregate readout", ""])
    for metric in metric_fields:
        item = summary[metric]
        lines.append(
            f"- `{metric}`: mean {item['mean']:+.6f}; positive {item['positive_count']}/{item['total']}."
        )
    lines.extend(["", "## Preregistered gate", ""])
    for gate, passed in summary["preregistered_readout"].items():
        lines.append(f"- `{gate}`: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This factorial determines whether the two frozen contributions coexist under one protocol. It does not authorize model tuning from consumed test results and does not establish cross-dataset periodic generality.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT)
    print(OUTPUT / "factorial_pairs.csv")
    print(OUTPUT / "factorial_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
