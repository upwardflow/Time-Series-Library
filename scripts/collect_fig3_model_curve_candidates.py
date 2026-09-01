#!/usr/bin/env python3
"""Export aligned model-wise forecast curves from formal frozen checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper/neurocomputing/figures/fig3_curve_candidates/source_npz"
FORMAL_RECORDS = ROOT / "logs/q2_main_baselines/records"
DATASETS = {
    "ETTh1": (ROOT / "dataset/ETT-small/ETTh1.csv", "OT"),
    "ETTh2": (ROOT / "dataset/ETT-small/ETTh2.csv", "OT"),
    "ETTm1": (ROOT / "dataset/ETT-small/ETTm1.csv", "OT"),
    "ETTm2": (ROOT / "dataset/ETT-small/ETTm2.csv", "OT"),
    "Weather": (ROOT / "dataset/weather/weather.csv", "CO2 (ppm)"),
}
HORIZONS = (96, 192, 336, 720)
MODELS = {
    "TimeRole": "graphmambacmrhm",
    "S-Mamba": "smamba",
    "iTransformer": "itransformer",
    "TimeMixer": "timemixer",
    "MSGNet": "msgnet",
    "PatchTST": "patchtst",
    "TimesNet": "timesnet",
    "DLinear": "dlinear",
}


def select_origin(data_path: Path, target: str, dataset: str) -> dict[str, object]:
    frame = pd.read_csv(data_path)
    values = frame[target].to_numpy(dtype=np.float64)
    if dataset.startswith("ETTm"):
        test_start = 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4
        test_length = 4 * 30 * 24 * 4
    elif dataset.startswith("ETTh"):
        test_start = 12 * 30 * 24 + 4 * 30 * 24
        test_length = 4 * 30 * 24
    else:
        test_length = int(len(frame) * 0.2)
        test_start = len(frame) - test_length
    test = values[test_start:test_start + test_length]
    windows = np.lib.stride_tricks.sliding_window_view(test, max(HORIZONS))
    variation = np.abs(np.diff(windows, axis=1)).sum(axis=1)
    median = float(np.median(variation))
    origin = int(np.argmin(np.abs(variation - median)))
    return {
        "dataset": dataset,
        "target": target,
        "selection_rule": "nearest_median_ground_truth_total_variation_h720",
        "origin": origin,
        "valid_origins_h720": int(len(variation)),
        "selected_total_variation": float(variation[origin]),
        "median_total_variation": median,
        "forecast_start_row": int(test_start + origin),
    }


def formal_record(dataset: str, horizon: int, model_slug: str) -> Path:
    name = f"{model_slug}_{dataset.lower()}_sl336_pl{horizon}_s2021.json"
    path = FORMAL_RECORDS / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def resolve_command(record_path: Path) -> tuple[list[str], list[str]]:
    chain: list[str] = []
    current = record_path
    while True:
        record = json.loads(current.read_text(encoding="utf-8"))
        chain.append(str(current))
        if record.get("command"):
            return list(record["command"]), chain
        source = record.get("source_record")
        if not source:
            raise ValueError(f"no command or source_record in {current}")
        current = Path(source)
        if not current.is_absolute():
            current = ROOT / current


def set_option(command: list[str], option: str, value: str) -> None:
    if option in command:
        command[command.index(option) + 1] = value
    else:
        command.extend([option, value])


def apply_timemixer_stability_override(
    command: list[str], dataset: str, horizon: int
) -> tuple[list[str], str]:
    command = list(command)
    if dataset in {"ETTm2", "Weather"}:
        slug = f"timemixer_stable_{dataset.lower()}_sl336_pl{horizon}_lr1e4_s2021"
        checkpoint_root = ROOT / "logs/q2_timemixer_stability/checkpoints"
        source = "validated_lr1e4_stability_repair"
    elif dataset == "ETTm1" and horizon in {96, 720}:
        slug = f"timemixer_ettm1_sl336_pl{horizon}_lr1e4_final_s2021"
        checkpoint_root = ROOT / "logs/q2_timemixer_ettm1_repair/checkpoints"
        source = "validated_lr1e4_ettm1_repair"
    else:
        return command, "formal_seed2021_record"
    set_option(command, "--model_id", slug)
    set_option(command, "--des", slug)
    set_option(command, "--learning_rate", "0.0001")
    set_option(command, "--checkpoints", str(checkpoint_root))
    return command, source


def evaluation_command(command: list[str], output: Path, origin: int) -> list[str]:
    command = list(command)
    command[0] = str(ROOT / ".venv/bin/python")
    command[command.index("--is_training") + 1] = "0"
    if "--checkpoints" not in command:
        command.extend(["--checkpoints", str(ROOT / "checkpoints")])
    command.extend([
        "--forecast_export_path", str(output),
        "--forecast_export_origin", str(origin),
        "--forecast_export_channel", "-1",
        "--forecast_export_context", "96",
        "--forecast_export_only",
    ])
    return command


def validate_exports(rows: list[dict[str, object]], expected_models: int) -> None:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), int(row["horizon"])), []).append(row)
    targets: dict[tuple[str, int], np.ndarray] = {}
    for key, group in grouped.items():
        if len(group) != expected_models:
            raise ValueError(f"expected {expected_models} models for {key}, found {len(group)}")
        loaded = [np.load(str(item["output"]), allow_pickle=False) for item in group]
        reference = loaded[0]["target"]
        for item in loaded:
            if not np.allclose(reference, item["target"], rtol=1e-6, atol=1e-5):
                raise ValueError(f"ground truth differs across models for {key}")
            if len(item["context"]) != 96 or len(item["prediction"]) != key[1]:
                raise ValueError(f"unexpected compact export shape for {key}")
        targets[key] = reference.copy()
        for item in loaded:
            item.close()
    exported_datasets = sorted({key[0] for key in targets})
    for dataset in exported_datasets:
        exported_horizons = sorted(key[1] for key in targets if key[0] == dataset)
        long_target = targets[(dataset, max(exported_horizons))]
        for horizon in exported_horizons:
            if not np.allclose(
                targets[(dataset, horizon)], long_target[:horizon],
                rtol=1e-6, atol=1e-5,
            ):
                raise ValueError(f"target prefix mismatch for {dataset} H={horizon}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--datasets", nargs="*", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--horizons", nargs="*", type=int, choices=HORIZONS, default=list(HORIZONS))
    parser.add_argument("--models", nargs="*", choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args()
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir.parent / "inference_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    selections = {
        dataset: select_origin(*DATASETS[dataset], dataset)
        for dataset in args.datasets
    }
    (output_dir / "selection_protocol.json").write_text(
        json.dumps(selections, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        origin = int(selections[dataset]["origin"])
        for horizon in args.horizons:
            for label in args.models:
                slug = MODELS[label]
                record_path = formal_record(dataset, horizon, slug)
                source_command, record_chain = resolve_command(record_path)
                if label == "TimeMixer":
                    source_command, command_source = apply_timemixer_stability_override(
                        source_command, dataset, horizon
                    )
                else:
                    command_source = "formal_seed2021_record"
                output = output_dir / f"{dataset.lower()}_h{horizon}_{slug}.npz"
                command = evaluation_command(source_command, output, origin)
                print(f"[{dataset} H={horizon} {label}] {output}", flush=True)
                if args.dry_run:
                    print(" ".join(command), flush=True)
                elif args.force or not output.exists():
                    log_path = log_dir / f"{dataset.lower()}_h{horizon}_{slug}.log"
                    with log_path.open("w", encoding="utf-8") as log:
                        subprocess.run(
                            command, cwd=ROOT, check=True,
                            stdout=log, stderr=subprocess.STDOUT,
                        )
                rows.append({
                    "dataset": dataset,
                    "horizon": horizon,
                    "model": label,
                    "model_slug": slug,
                    "origin": origin,
                    "target": DATASETS[dataset][1],
                    "output": str(output),
                    "formal_record": str(record_path),
                    "command_source": command_source,
                    "record_chain": json.dumps(record_chain),
                    "command": json.dumps(command),
                })

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not args.dry_run:
        validate_exports(rows, expected_models=len(args.models))
        print(f"validated_exports={len(rows)}")


if __name__ == "__main__":
    main()
