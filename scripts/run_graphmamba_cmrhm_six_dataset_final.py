#!/usr/bin/env python3
"""Overnight final-test matrix for the frozen GraphMambaCMRHM-v1 model.

ETTm1/ETTm2 results are imported from their completed one-shot test records.
ETTh1, ETTh2, weather, and solar are trained with validation checkpoint
selection and evaluated on test exactly once per horizon. Progress artifacts
are rewritten after every completed run so the matrix is safely resumable.
"""

from __future__ import annotations

import argparse
import csv
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


ROOT = Path(__file__).resolve().parents[1]
RUN_PY = ROOT / "run.py"
OUTPUT = ROOT / "logs" / "graphmamba_cmrhm_six_dataset_final"
REPORT = ROOT / "experiment_results" / "GraphMamba_CMRHM_six_dataset_final_test.md"
EXISTING_ETTM = ROOT / "logs" / "graphmamba_cmrhm_final_test" / "records"
PRED_LENS = (96, 192, 336, 720)
NEW_DATASETS = ("ETTh1", "ETTh2", "weather", "solar")
ALL_DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "solar")
SEED = 2021
VALIDATION_PATTERN = re.compile(r"^VALIDATION_RESULT\s+(\{.*\})\s*$")
TEST_PATTERN = re.compile(r"^mse:([-+0-9.eE]+),\s*mae:([-+0-9.eE]+),\s*dtw:")
SETTING_PATTERN = re.compile(r"^>+start training : (.*?)>+$")


@dataclass(frozen=True)
class DatasetConfig:
    root_path: Path
    data_path: str
    data_type: str
    channels: int
    target: str
    batch_size: int


DATASETS = {
    "ETTh1": DatasetConfig(ROOT / "dataset" / "ETT-small", "ETTh1.csv", "ETTh1", 7, "OT", 32),
    "ETTh2": DatasetConfig(ROOT / "dataset" / "ETT-small", "ETTh2.csv", "ETTh2", 7, "OT", 32),
    "weather": DatasetConfig(ROOT / "dataset" / "weather", "weather.csv", "custom", 21, "CO2 (ppm)", 32),
    "solar": DatasetConfig(ROOT / "dataset" / "solar", "solar.csv", "custom", 137, "channel_99", 16),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_name(dataset: str, pred_len: int) -> str:
    return f"sixds_{dataset.lower()}_p{pred_len}_s{SEED}"


def record_path(dataset: str, pred_len: int) -> Path:
    return OUTPUT / "records" / f"{run_name(dataset, pred_len)}.json"


def completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "completed" and "test_mse" in payload


def validate_data() -> None:
    failures = []
    for dataset in NEW_DATASETS:
        config = DATASETS[dataset]
        path = config.root_path / config.data_path
        if not path.is_file():
            failures.append(f"missing dataset: {path}")
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            columns = next(csv.reader(handle), [])
        values = [column for column in columns if column != "date"]
        if "date" not in columns:
            failures.append(f"{path}: missing date column")
        if len(values) != config.channels:
            failures.append(
                f"{path}: expected {config.channels} variables, found {len(values)}"
            )
        if config.target not in values:
            failures.append(f"{path}: target {config.target!r} not found")
    if failures:
        raise RuntimeError("\n".join(failures))


def build_command(dataset: str, pred_len: int, args: argparse.Namespace) -> list[str]:
    config = DATASETS[dataset]
    candidate = run_name(dataset, pred_len)
    return [
        sys.executable,
        "-u",
        str(RUN_PY),
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--root_path", str(config.root_path),
        "--data_path", config.data_path,
        "--model_id", f"{dataset}_336_{pred_len}_{candidate}",
        "--model", "GraphMambaCMRHM",
        "--seed", str(SEED),
        "--data", config.data_type,
        "--features", "M",
        "--target", config.target,
        "--seq_len", "336",
        "--label_len", "48",
        "--pred_len", str(pred_len),
        "--enc_in", str(config.channels),
        "--dec_in", str(config.channels),
        "--c_out", str(config.channels),
        "--patch_len", "4",
        "--stride", "2",
        "--d_model", "64",
        "--d_ff", "128",
        "--d_state", "32",
        "--d_conv", "2",
        "--e_layers", "1",
        "--expand", "2",
        "--mamba_version", "1",
        "--mamba_bidirectional", "1",
        "--use_graph", "1",
        "--use_time_mamba", "1",
        "--use_patch", "1",
        "--use_decomp", "1",
        "--moving_avg", "25",
        "--dual_scale_scan_mode", "auto",
        # Required when GraphMambaCMRHM's recent-96 backbone resolves hourly
        # ETT to periodic_aligned. They are ignored by non-periodic datasets.
        "--periodic_period", "24",
        "--periodic_local_patch", "4",
        "--periodic_local_stride", "2",
        "--periodic_period_stride", "12",
        "--periodic_use_adapter", "1",
        "--graph_alpha", "0.5",
        "--graph_top_k", "2",
        "--graph_sample_size", "2000",
        "--graph_sample_method", "uniform",
        "--static_graph_mode", "weighted",
        "--graph_cache", "1",
        "--dropout", "0.1",
        "--batch_size", str(config.batch_size),
        "--learning_rate", "0.0005",
        "--lradj", "type1",
        "--train_epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--num_workers", "0",
        "--gpu", "0",
        "--des", candidate,
        "--itr", "1",
        "--test_after_train", "1",
    ]


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_one(dataset: str, pred_len: int, args: argparse.Namespace) -> int:
    destination = record_path(dataset, pred_len)
    if args.resume and completed(destination):
        print(f"Already completed: {destination}", flush=True)
        return 0

    command = build_command(dataset, pred_len, args)
    log_path = OUTPUT / "logs" / f"{run_name(dataset, pred_len)}.log"
    print("Command:", shlex.join(command), flush=True)
    print("Log:", log_path, flush=True)
    if args.dry_run:
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    validation = None
    metrics = None
    setting = None
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
            stripped = line.strip()
            if match := VALIDATION_PATTERN.match(stripped):
                validation = json.loads(match.group(1))
            if match := TEST_PATTERN.match(stripped):
                metrics = {
                    "test_mse": float(match.group(1)),
                    "test_mae": float(match.group(2)),
                }
            if match := SETTING_PATTERN.match(stripped):
                setting = match.group(1)
        return_code = process.wait()

    payload: dict[str, object] = {
        "status": "completed" if return_code == 0 and validation and metrics else "failed",
        "dataset": dataset,
        "pred_len": pred_len,
        "model": "GraphMambaCMRHM",
        "seed": SEED,
        "seq_len": 336,
        "backbone_recent_len": 96,
        "test_access": "one_shot_after_validation_selection",
        "checkpoint_selected_by": "validation_best_mse",
        "periodic_backbone_active": dataset in {"ETTh1", "ETTh2"},
        "return_code": return_code,
        "duration_seconds": round(time.monotonic() - started, 3),
        "recorded_at": datetime.now().astimezone().isoformat(),
        "command": command,
        "setting": setting,
        "checkpoint": str(ROOT / "checkpoints" / setting / "checkpoint.pth") if setting else None,
        "log_path": str(log_path),
        "result_source": "new_overnight_run",
    }
    if validation:
        payload.update({f"validation_{key}": value for key, value in validation.items()})
    if metrics:
        payload.update(metrics)
    atomic_json(destination, payload)
    return 0 if payload["status"] == "completed" else return_code or 1


def existing_ettm_record(dataset: str, pred_len: int) -> dict[str, object] | None:
    source = EXISTING_ETTM / f"{dataset.lower()}_{pred_len}_cmrhm_s{SEED}.json"
    if not completed(source):
        return None
    payload = json.loads(source.read_text(encoding="utf-8"))
    return {
        "dataset": dataset,
        "pred_len": pred_len,
        "seed": SEED,
        "model": "GraphMambaCMRHM",
        "periodic_backbone_active": False,
        "test_mse": payload["test_mse"],
        "test_mae": payload["test_mae"],
        "validation_best_mse": payload.get("validation_best_mse"),
        "validation_best_mae": payload.get("validation_best_mae"),
        "result_source": "reused_existing_one_shot_test",
        "source_record": str(source),
    }


def collect_rows() -> list[dict[str, object]]:
    rows = []
    for dataset in ALL_DATASETS:
        for pred_len in PRED_LENS:
            if dataset in {"ETTm1", "ETTm2"}:
                record = existing_ettm_record(dataset, pred_len)
            else:
                path = record_path(dataset, pred_len)
                record = json.loads(path.read_text(encoding="utf-8")) if completed(path) else None
            if record:
                rows.append({
                    "dataset": dataset,
                    "pred_len": pred_len,
                    "seed": SEED,
                    "model": "GraphMambaCMRHM",
                    "periodic_backbone_active": record["periodic_backbone_active"],
                    "validation_mse": record.get("validation_best_mse"),
                    "validation_mae": record.get("validation_best_mae"),
                    "test_mse": record["test_mse"],
                    "test_mae": record["test_mae"],
                    "result_source": record["result_source"],
                })
    return rows


def write_outputs(rows: list[dict[str, object]], current: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT / "results.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    completed_new = sum(row["result_source"] == "new_overnight_run" for row in rows)
    status = {
        "status": "completed" if len(rows) == 24 else "running",
        "current": current,
        "completed_total": len(rows),
        "expected_total": 24,
        "completed_new": completed_new,
        "expected_new": 16,
        "reused_existing": len(rows) - completed_new,
        "updated_at": datetime.now().astimezone().isoformat(),
        "results_csv": str(csv_path),
        "report": str(REPORT),
    }
    atomic_json(OUTPUT / "status.json", status)

    lines = [
        "# GraphMambaCMRHM 六数据集正式测试",
        "",
        f"- 状态：**{status['status']}**（{len(rows)}/24）",
        f"- 当前任务：`{current}`",
        "- 模型：冻结的 `GraphMambaCMRHM-v1`，输入长度 336，主干最近窗口 96",
        "- 数据集：ETTh1、ETTh2、ETTm1、ETTm2、Weather、Solar；排除 Traffic/ECL",
        "- 预测长度：96、192、336、720；seed 2021",
        "- 协议：训练集优化、验证集最佳 MSE 选 checkpoint、测试集一次性评估",
        "- ETTm1/ETTm2 使用既有一次性正式测试，不重复访问测试集",
        "- ETTh1/ETTh2 启用周期 24 双尺度主干；其余数据使用独立双 Patch 主干",
        "",
        "| Dataset | Horizon | Periodic backbone | Validation MSE | Test MSE | Test MAE | Source |",
        "|---|---:|:---:|---:|---:|---:|---|",
    ]
    for row in rows:
        validation = row["validation_mse"]
        validation_text = f"{validation:.6f}" if isinstance(validation, (int, float)) else "—"
        source = "本轮新增" if row["result_source"] == "new_overnight_run" else "既有正式结果"
        lines.append(
            f"| {row['dataset']} | {row['pred_len']} | "
            f"{'yes' if row['periodic_backbone_active'] else 'no'} | "
            f"{validation_text} | {row['test_mse']:.6f} | {row['test_mae']:.6f} | {source} |"
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "这张表报告冻结模型的绝对结果，不使用测试结果修改结构或超参数。只有 ETTh1/ETTh2 "
        "可称为周期双尺度 + CMRHM；ETTm1/ETTm2、Weather、Solar 是独立双 Patch + CMRHM。",
        "",
        f"机器可读汇总：`{csv_path.relative_to(ROOT)}`。",
        "",
    ])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    validate_data()
    rows = collect_rows()
    write_outputs(rows, "initializing")
    failures = []
    for dataset in NEW_DATASETS:
        for pred_len in PRED_LENS:
            current = f"{dataset}-{pred_len}"
            print(f"\n=== FINAL MATRIX {current} ===", flush=True)
            return_code = run_one(dataset, pred_len, args)
            if args.dry_run:
                continue
            rows = collect_rows()
            write_outputs(rows, current)
            if return_code:
                failures.append({"task": current, "return_code": return_code})
                atomic_json(OUTPUT / "failures.json", {"failures": failures})
                print(f"Failed: {current}; continuing overnight matrix", file=sys.stderr, flush=True)
    if args.dry_run:
        print("Dry run complete; no training or test split was accessed.", flush=True)
        return 0
    rows = collect_rows()
    final_state = "complete" if len(rows) == 24 and not failures else "finished_with_failures"
    write_outputs(rows, final_state)
    print(f"Matrix finished: {len(rows)}/24 records; failures={len(failures)}", flush=True)
    return 0 if final_state == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
