#!/usr/bin/env python3
"""Export compact real forecast profiles from existing frozen checkpoints.

The origin selection uses ground truth only: for each dataset, choose the valid
H=720 test window whose OT total variation is nearest the dataset median. The
same origin is then reused for every model and for H=96/H=720.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper/neurocomputing/figures/forecast_profiles/source_npz"
HORIZONS = (96, 720)


def select_median_variation_origin(data_path: Path, dataset: str) -> dict[str, float | int | str]:
    frame = pd.read_csv(data_path)
    if "OT" not in frame.columns:
        raise ValueError(f"{data_path} does not contain the required OT channel")
    samples_per_hour = 4 if dataset.startswith("ETTm") else 1
    train_length = 12 * 30 * 24 * samples_per_hour
    split_length = 4 * 30 * 24 * samples_per_hour
    test_start = train_length + split_length
    test = frame["OT"].to_numpy(dtype=np.float64)[test_start:test_start + split_length]
    windows = np.lib.stride_tricks.sliding_window_view(test, 720)
    variation = np.abs(np.diff(windows, axis=1)).sum(axis=1)
    median = float(np.median(variation))
    origin = int(np.argmin(np.abs(variation - median)))
    return {
        "dataset": dataset,
        "channel": "OT",
        "selection_rule": "nearest_median_ground_truth_total_variation_h720",
        "origin": origin,
        "valid_origins_h720": int(len(variation)),
        "selected_total_variation": float(variation[origin]),
        "median_total_variation": median,
        "test_start_row": int(test_start),
        "forecast_start_row": int(test_start + origin),
    }


def load_record_command(path: Path) -> list[str]:
    record = json.loads(path.read_text(encoding="utf-8"))
    command = list(record["command"])
    model_index = command.index("--model") + 1
    command[model_index] = {
        "GraphMambaCMRHM": "TimeRole",
        "CMRHM": "TimeRole",
        "GraphMambaRecent": "TimeRoleRecent",
        "GraphMamba": "TimeRoleFullHistory",
    }.get(command[model_index], command[model_index])
    training_index = command.index("--is_training") + 1
    command[training_index] = "0"
    return command


def dlinear_command(horizon: int) -> list[str]:
    return [
        str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "run.py"),
        "--task_name", "long_term_forecast", "--is_training", "0",
        "--root_path", str(ROOT / "dataset/ETT-small"),
        "--data_path", "ETTh1.csv", "--model_id", f"ETTh1_96_{horizon}",
        "--model", "DLinear", "--seed", "2021", "--data", "ETTh1",
        "--features", "M", "--target", "OT", "--seq_len", "96",
        "--label_len", "48", "--pred_len", str(horizon), "--e_layers", "2",
        "--d_layers", "1", "--factor", "3", "--enc_in", "7",
        "--dec_in", "7", "--c_out", "7", "--des", "Exp", "--itr", "1",
        "--num_workers", "0", "--checkpoints", str(ROOT / "checkpoints"),
    ]


def experiment_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for horizon in HORIZONS:
        specs.extend(
            [
                {
                    "dataset": "ETTm1",
                    "horizon": horizon,
                    "label": "TimeRole",
                    "command": load_record_command(
                        ROOT / "logs/graphmamba_cmrhm_final_test/records"
                        / f"ettm1_{horizon}_cmrhm_s2021.json"
                    ),
                },
                {
                    "dataset": "ETTm1",
                    "horizon": horizon,
                    "label": "RGSP-96",
                    "command": load_record_command(
                        ROOT / "logs/graphmamba_cmrhm_final_test/records"
                        / f"ettm1_{horizon}_recent336_s2021.json"
                    ),
                },
                {
                    "dataset": "ETTh1",
                    "horizon": horizon,
                    "label": "TimeRole",
                    "command": load_record_command(
                        ROOT / "logs/timerole_table2_multiseed/records"
                        / f"timerole_etth1_sl336_pl{horizon}_s2022.json"
                    ),
                },
                {
                    "dataset": "ETTh1",
                    "horizon": horizon,
                    "label": "DLinear",
                    "command": dlinear_command(horizon),
                },
            ]
        )
    return specs


def append_export_args(command: list[str], output: Path, origin: int) -> list[str]:
    return command + [
        "--forecast_export_path", str(output),
        "--forecast_export_origin", str(origin),
        "--forecast_export_channel", "-1",
        "--forecast_export_context", "96",
        "--forecast_export_inverse",
    ]


def validate_exports(rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), int(row["horizon"])), []).append(row)

    targets: dict[tuple[str, int], np.ndarray] = {}
    for key, group in grouped.items():
        loaded = [np.load(Path(str(item["output"])), allow_pickle=False) for item in group]
        reference = loaded[0]["target"]
        for item in loaded[1:]:
            if not np.allclose(reference, item["target"], rtol=1e-6, atol=1e-5):
                raise ValueError(f"ground truth differs across models for {key}")
        targets[key] = reference.copy()
        for item in loaded:
            if len(item["context"]) != 96 or len(item["prediction"]) != key[1]:
                raise ValueError(f"unexpected compact export shape for {key}")

    for dataset in {key[0] for key in targets}:
        if not np.allclose(
            targets[(dataset, 96)], targets[(dataset, 720)][:96],
            rtol=1e-6, atol=1e-5,
        ):
            raise ValueError(f"H=96 target is not the H=720 prefix for {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="rerun exports that already exist")
    parser.add_argument("--dry-run", action="store_true", help="print commands without inference")
    args = parser.parse_args()
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    selections = {
        "ETTm1": select_median_variation_origin(
            ROOT / "dataset/ETT-small/ETTm1.csv", "ETTm1"
        ),
        "ETTh1": select_median_variation_origin(
            ROOT / "dataset/ETT-small/ETTh1.csv", "ETTh1"
        ),
    }
    (output_dir / "selection_protocol.json").write_text(
        json.dumps(selections, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    for spec in experiment_specs():
        dataset = str(spec["dataset"])
        horizon = int(spec["horizon"])
        label = str(spec["label"])
        slug = label.lower().replace("-", "_")
        output = output_dir / f"{dataset.lower()}_h{horizon}_{slug}.npz"
        command = append_export_args(
            list(spec["command"]), output, int(selections[dataset]["origin"])
        )
        print(f"[{dataset} H={horizon} {label}] {output}", flush=True)
        if args.dry_run:
            print(" ".join(command), flush=True)
        elif args.force or not output.exists():
            subprocess.run(command, cwd=ROOT, check=True)
        rows.append({
            "dataset": dataset,
            "horizon": horizon,
            "model": label,
            "origin": int(selections[dataset]["origin"]),
            "output": str(output),
            "command": json.dumps(command),
        })

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not args.dry_run:
        validate_exports(rows)
        print(f"validated_exports={len(rows)}")
        print(f"selection_ETTm1={selections['ETTm1']['origin']}")
        print(f"selection_ETTh1={selections['ETTh1']['origin']}")


if __name__ == "__main__":
    main()
