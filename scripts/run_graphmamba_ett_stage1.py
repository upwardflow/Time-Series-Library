#!/usr/bin/env python3
"""Run validation-only GraphMamba ETT stage-1 searches.

This runner deliberately passes ``--test_after_train 0``.  Run one stage,
let the results be reviewed, then use the selected JSON configuration as the
base for the next stage.

Examples:
    python scripts/run_graphmamba_ett_stage1.py --stage baseline --dry-run
    python scripts/run_graphmamba_ett_stage1.py --stage baseline
    python scripts/run_graphmamba_ett_stage1.py --stage optimizer \
        --base-config logs/graphmamba_ett_stage1/selected/baseline.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_PY = REPO_ROOT / "run.py"
DATA_ROOT = REPO_ROOT / "dataset" / "ETT-small"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "graphmamba_ett_stage1_weighted"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
PRED_LENS = (96, 192, 336, 720)
ANCHOR_PRED_LENS = (96, 720)
VALIDATION_PATTERN = re.compile(r"^VALIDATION_RESULT\s+(\{.*\})\s*$")


@dataclass(frozen=True)
class Candidate:
    name: str
    config: dict[str, Any]


BASE_CONFIG: dict[str, Any] = {
    "learning_rate": 5e-4,
    "lradj": "type1",
    "batch_size": 32,
    "dropout": 0.1,
    "patch_len": 4,
    "stride": 2,
    "moving_avg": 25,
    "graph_alpha": 0.5,
    "graph_top_k": 2,
    "node_dim": 10,
    "static_graph_mode": "weighted",
    "d_model": 64,
    "d_ff": 128,
    "d_state": 32,
    "d_conv": 2,
    "e_layers": 1,
    "expand": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-only staged search for GraphMamba on ETT."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("baseline", "optimizer", "multiscale", "graph", "capacity"),
    )
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--pred-lens", nargs="+", type=int, choices=PRED_LENS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--gpu", type=int, default=0, help="physical CUDA GPU index")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-runs", type=int, help="limit runs for smoke testing")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.patience < 1:
        parser.error("--epochs and --patience must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.max_runs is not None and args.max_runs < 1:
        parser.error("--max-runs must be positive")
    if args.stage != "baseline" and args.base_config is None:
        parser.error("non-baseline stages require --base-config selected after review")

    args.datasets = list(dict.fromkeys(args.datasets))
    default_lens = PRED_LENS if args.stage == "baseline" else ANCHOR_PRED_LENS
    args.pred_lens = list(dict.fromkeys(args.pred_lens or default_lens))
    args.output_dir = args.output_dir.resolve()
    if args.base_config is not None:
        args.base_config = args.base_config.resolve()
    return args


def load_base_config(path: Path | None) -> dict[str, Any]:
    config = dict(BASE_CONFIG)
    if path is None:
        return config
    with path.open("r", encoding="utf-8") as handle:
        supplied = json.load(handle)
    if "config" in supplied:
        supplied = supplied["config"]
    elif "base_config" in supplied:
        supplied = supplied["base_config"]
    unknown = sorted(set(supplied) - set(BASE_CONFIG))
    if unknown:
        raise ValueError(f"unknown base configuration keys: {unknown}")
    config.update(supplied)
    return config


def changed(base: dict[str, Any], **updates: Any) -> dict[str, Any]:
    config = dict(base)
    config.update(updates)
    if "d_model" in updates and "d_ff" not in updates:
        config["d_ff"] = 2 * int(config["d_model"])
    return config


def build_candidates(stage: str, base: dict[str, Any]) -> list[Candidate]:
    if stage == "baseline":
        raw = [("baseline", base)]
    elif stage == "optimizer":
        raw = [("base", base)]
        raw += [(f"lr_{value:g}", changed(base, learning_rate=value))
                for value in (1e-4, 3e-4, 1e-3)]
        raw += [(f"batch_{value}", changed(base, batch_size=value))
                for value in (16, 64)]
        raw += [(f"dropout_{value:g}", changed(base, dropout=value))
                for value in (0.0, 0.2)]
        raw += [(f"lradj_{value}", changed(base, lradj=value))
                for value in ("type3", "cosine")]
    elif stage == "multiscale":
        raw = [("base", base)]
        raw += [
            (f"patch_{patch}_stride_{stride}", changed(base, patch_len=patch, stride=stride))
            for patch, stride in ((8, 4), (16, 4), (16, 8))
        ]
        raw += [(f"moving_avg_{value}", changed(base, moving_avg=value))
                for value in (13, 49)]
    elif stage == "graph":
        raw = [("base", base)]
        raw += [(f"alpha_{value:g}", changed(base, graph_alpha=value))
                for value in (0.0, 0.25, 0.75, 1.0)]
        raw += [(f"topk_{value}", changed(base, graph_top_k=value))
                for value in (1, 3, 4)]
        raw += [(f"node_dim_{value}", changed(base, node_dim=value))
                for value in (8, 16, 32)]
        raw += [("binary_graph", changed(base, static_graph_mode="binary"))]
    elif stage == "capacity":
        raw = [("base", base)]
        raw += [(f"d_model_{value}", changed(base, d_model=value))
                for value in (32, 128)]
        raw += [(f"d_state_{value}", changed(base, d_state=value))
                for value in (16, 64)]
        raw += [
            ("d_conv_4", changed(base, d_conv=4)),
            ("e_layers_2", changed(base, e_layers=2)),
        ]
    else:
        raise ValueError(f"unsupported stage: {stage}")

    candidates = []
    seen = set()
    for name, config in raw:
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
        if canonical in seen:
            continue
        seen.add(canonical)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
        candidates.append(Candidate(f"{name}_{digest}", config))
    return candidates


def build_command(
    args: argparse.Namespace,
    dataset: str,
    pred_len: int,
    candidate: Candidate,
) -> list[str]:
    config = candidate.config
    description = f"ett_s1_{args.stage}_{candidate.name}"
    return [
        sys.executable, "-u", str(RUN_PY),
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--root_path", str(DATA_ROOT),
        "--data_path", f"{dataset}.csv",
        "--model_id", f"{dataset}_96_{pred_len}_{candidate.name}",
        "--model", "GraphMamba",
        "--seed", str(args.seed),
        "--data", dataset,
        "--features", "M",
        "--target", "OT",
        "--seq_len", "96",
        "--label_len", "48",
        "--pred_len", str(pred_len),
        "--enc_in", "7", "--dec_in", "7", "--c_out", "7",
        "--patch_len", str(config["patch_len"]),
        "--stride", str(config["stride"]),
        "--d_model", str(config["d_model"]),
        "--d_ff", str(config["d_ff"]),
        "--d_state", str(config["d_state"]),
        "--d_conv", str(config["d_conv"]),
        "--e_layers", str(config["e_layers"]),
        "--expand", str(config["expand"]),
        "--mamba_version", "1",
        "--mamba_bidirectional", "1",
        "--use_graph", "1", "--use_time_mamba", "1",
        "--use_patch", "1", "--use_decomp", "1",
        "--moving_avg", str(config["moving_avg"]),
        "--graph_alpha", str(config["graph_alpha"]),
        "--graph_top_k", str(config["graph_top_k"]),
        "--node_dim", str(config["node_dim"]),
        "--graph_sample_size", "2000",
        "--graph_sample_method", "uniform",
        "--static_graph_mode", str(config["static_graph_mode"]),
        "--graph_cache", "0",
        "--dropout", str(config["dropout"]),
        "--batch_size", str(config["batch_size"]),
        "--learning_rate", str(config["learning_rate"]),
        "--lradj", str(config["lradj"]),
        "--train_epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--num_workers", str(args.num_workers),
        "--gpu", "0",
        "--des", description,
        "--itr", "1",
        "--test_after_train", "0",
    ]


def run_process(command: list[str], log_path: Path, env: dict[str, str]) -> tuple[int, dict[str, Any] | None]:
    result = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
            match = VALIDATION_PATTERN.match(line.strip())
            if match:
                result = json.loads(match.group(1))
        return process.wait(), result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def rebuild_summaries(stage_dir: Path, baseline_dir: Path) -> None:
    records = []
    for path in sorted((stage_dir / "records").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            records.append(payload)

    baseline = {}
    for path in sorted((baseline_dir / "records").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "completed":
            baseline[(payload["dataset"], payload["pred_len"])] = payload["best_mse"]

    run_fields = [
        "stage", "candidate", "dataset", "pred_len", "seed", "best_mse", "best_mae",
        "best_epoch", "epochs_ran", "duration_seconds", "recorded_at", "log_path",
    ]
    with (stage_dir / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in run_fields})

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["candidate"], []).append(record)
    summary_fields = [
        "candidate", "completed_runs", "mean_val_mse", "mean_val_mae",
        "normalized_val_mse",
    ]
    with (stage_dir / "candidate_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for candidate, items in sorted(grouped.items()):
            mean_mse = sum(item["best_mse"] for item in items) / len(items)
            mean_mae = sum(item["best_mae"] for item in items) / len(items)
            ratios = [
                item["best_mse"] / baseline[(item["dataset"], item["pred_len"])]
                for item in items
                if (item["dataset"], item["pred_len"]) in baseline
            ]
            writer.writerow({
                "candidate": candidate,
                "completed_runs": len(items),
                "mean_val_mse": f"{mean_mse:.10f}",
                "mean_val_mae": f"{mean_mae:.10f}",
                "normalized_val_mse": f"{sum(ratios) / len(ratios):.10f}" if ratios else "",
            })


def main() -> int:
    args = parse_args()
    missing = [str(DATA_ROOT / f"{dataset}.csv") for dataset in args.datasets
               if not (DATA_ROOT / f"{dataset}.csv").is_file()]
    if missing:
        print("Missing datasets:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 2

    try:
        base = load_base_config(args.base_config)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Cannot load base configuration: {error}", file=sys.stderr)
        return 2
    candidates = build_candidates(args.stage, base)
    stage_dir = args.output_dir / args.stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    dataset_tag = "_".join(args.datasets)
    write_json(stage_dir / f"candidate_configs_{dataset_tag}.json", {
        "stage": args.stage,
        "base_config": base,
        "candidates": [{"candidate": item.name, "config": item.config} for item in candidates],
    })

    jobs = [
        (candidate, dataset, pred_len)
        for candidate in candidates
        for dataset in args.datasets
        for pred_len in args.pred_lens
    ]
    if args.max_runs is not None:
        jobs = jobs[:args.max_runs]
    print(f"Stage {args.stage}: {len(candidates)} candidates, {len(jobs)} runs")
    print("Protocol: validation-only; the test split will not be evaluated")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    failures = []
    for index, (candidate, dataset, pred_len) in enumerate(jobs, start=1):
        run_id = f"{candidate.name}_{dataset}_96_{pred_len}_seed{args.seed}"
        record_path = stage_dir / "records" / f"{run_id}.json"
        log_path = stage_dir / "logs" / f"{run_id}.log"
        command = build_command(args, dataset, pred_len, candidate)
        print(f"\n[{index}/{len(jobs)}] {run_id}")
        if not args.no_resume and record_path.exists():
            previous = json.loads(record_path.read_text(encoding="utf-8"))
            if previous.get("status") == "completed":
                print("Completed record exists; skipping")
                continue
        print("Command:", shlex.join(command))
        if args.dry_run:
            continue

        started = time.monotonic()
        return_code, validation = run_process(command, log_path, env)
        duration = time.monotonic() - started
        status = "completed" if return_code == 0 and validation is not None else "failed"
        payload = {
            "status": status,
            "stage": args.stage,
            "candidate": candidate.name,
            "config": candidate.config,
            "dataset": dataset,
            "pred_len": pred_len,
            "seed": args.seed,
            "return_code": return_code,
            "duration_seconds": round(duration, 3),
            "recorded_at": datetime.now().astimezone().isoformat(),
            "log_path": str(log_path),
            "command": command,
        }
        if validation:
            payload.update(validation)
        write_json(record_path, payload)
        rebuild_summaries(stage_dir, args.output_dir / "baseline")
        if status != "completed":
            failures.append(run_id)
            print(f"Failed or missing validation marker: {run_id}", file=sys.stderr)
            if not args.continue_on_error:
                return return_code or 1

    if not args.dry_run:
        rebuild_summaries(stage_dir, args.output_dir / "baseline")
        print(f"\nStage records: {stage_dir / 'runs.csv'}")
        print(f"Candidate summary: {stage_dir / 'candidate_summary.csv'}")
    if failures:
        print(f"Completed with {len(failures)} failed run(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
