#!/usr/bin/env python3
"""Audit the frozen TimeRole simplification Phase C and apply all nine gates."""

from __future__ import annotations

import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "logs" / "timerole_recent_simplification"
PHASE_B_RECORDS = OUTPUT / "records" / "phase_b"
PHASE_C_RECORDS = OUTPUT / "records" / "phase_c"
GATE_PATH = OUTPUT / "audit" / "phase_c_gate.json"
RANKING_PATH = OUTPUT / "summaries" / "phase_c_candidate_ranking.csv"
REPORT_PATH = ROOT / "experiment_results" / "TimeRole_recent_predictor_phase_c_result.md"

ROLES = ("timerole", "recent")
VARIANTS = ("R0", "R2", "R4")
DATASETS = ("ETTm1", "ETTh2", "weather")
HORIZONS = (96, 720)
SEEDS = (2021, 2022, 2023)
NEW_SEEDS = (2022, 2023)
CORE_SOURCE_PATHS = (
    "data_provider/data_factory.py",
    "models/GraphMamba.py",
    "models/GraphMambaRecent.py",
    "models/TimeRole.py",
    "exp/exp_long_term_forecasting.py",
    "run.py",
)
PAIR_CONFIG_EXCLUSIONS = {"model", "model_id", "des"}
RESOURCE_KEYS = (
    "parameter_count",
    "milliseconds_per_batch",
    "train_peak_cuda_memory_bytes",
    "peak_cuda_memory_bytes",
)


def now() -> str:
    return datetime.now().astimezone().isoformat()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: object) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def load_records(directory: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not directory.is_dir():
        return rows
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cannot read record {path}: {error}") from error
        if not isinstance(payload, dict):
            raise RuntimeError(f"record is not a JSON object: {path}")
        payload["record_path"] = str(path)
        rows.append(payload)
    return rows


def key(row: dict[str, object]) -> tuple[str, str, int, int, str]:
    return (
        str(row.get("role")),
        str(row.get("dataset")),
        int(row.get("horizon", -1)),
        int(row.get("seed", -1)),
        str(row.get("variant")),
    )


def expected_keys() -> set[tuple[str, str, int, int, str]]:
    return {
        (role, dataset, horizon, seed, variant)
        for role in ROLES
        for dataset in DATASETS
        for horizon in HORIZONS
        for seed in SEEDS
        for variant in VARIANTS
    }


def selected_records() -> tuple[list[dict[str, object]], dict[str, object]]:
    phase_b_all = load_records(PHASE_B_RECORDS)
    phase_c_all = load_records(PHASE_C_RECORDS)
    phase_b = [
        row for row in phase_b_all
        if int(row.get("seed", -1)) == 2021
        and str(row.get("role")) in ROLES
        and str(row.get("variant")) in VARIANTS
        and str(row.get("dataset")) in DATASETS
        and int(row.get("horizon", -1)) in HORIZONS
    ]
    phase_c = [
        row for row in phase_c_all
        if int(row.get("seed", -1)) in NEW_SEEDS
        and str(row.get("role")) in ROLES
        and str(row.get("variant")) in VARIANTS
        and str(row.get("dataset")) in DATASETS
        and int(row.get("horizon", -1)) in HORIZONS
    ]
    metadata = {
        "phase_b_reused_records": len(phase_b),
        "phase_c_new_records": len(phase_c),
        "phase_b_other_records_ignored": len(phase_b_all) - len(phase_b),
        "phase_c_other_records_ignored": len(phase_c_all) - len(phase_c),
    }
    return phase_b + phase_c, metadata


def validate_completeness(
    rows: list[dict[str, object]], metadata: dict[str, object]
) -> dict[tuple[str, str, int, int, str], dict[str, object]]:
    observed: dict[tuple[str, str, int, int, str], dict[str, object]] = {}
    duplicates = []
    for row in rows:
        record_key = key(row)
        if record_key in observed:
            duplicates.append(record_key)
        observed[record_key] = row
    expected = expected_keys()
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    if missing or extra or duplicates:
        status = {
            "status": "incomplete",
            "expected_records": len(expected),
            "observed_unique_records": len(observed),
            "missing_count": len(missing),
            "missing_preview": missing[:20],
            "extra_count": len(extra),
            "extra_preview": extra[:20],
            "duplicate_count": len(duplicates),
            "duplicate_preview": duplicates[:20],
            **metadata,
        }
        raise RuntimeError(json.dumps(status, ensure_ascii=False, indent=2))
    return observed


def finite(row: dict[str, object], field: str) -> bool:
    try:
        return math.isfinite(float(row[field]))
    except (KeyError, TypeError, ValueError):
        return False


def validate_integrity(
    observed: dict[tuple[str, str, int, int, str], dict[str, object]]
) -> dict[str, object]:
    errors: list[str] = []
    for record_key, row in observed.items():
        if row.get("status") != "completed":
            errors.append(f"{record_key}: status={row.get('status')}")
        if int(row.get("return_code", -1)) != 0:
            errors.append(f"{record_key}: return_code={row.get('return_code')}")
        if row.get("split") != "val" or row.get("test_accessed") is not False:
            errors.append(f"{record_key}: invalid split/test provenance")
        if row.get("source_dirty") is not False:
            errors.append(f"{record_key}: source_dirty is not false")
        if row.get("validation_shuffle") is not False:
            errors.append(f"{record_key}: validation loader was shuffled")
        if int(row.get("data_order_seed", -1)) != record_key[3]:
            errors.append(f"{record_key}: data-order seed mismatch")
        for field in ("mse", "mae", *RESOURCE_KEYS):
            if not finite(row, field):
                errors.append(f"{record_key}: missing/non-finite {field}")
        if not Path(str(row.get("log_path", ""))).is_file():
            errors.append(f"{record_key}: raw log is missing")

    core_hashes: dict[str, set[str]] = {path: set() for path in CORE_SOURCE_PATHS}
    for row in observed.values():
        hashes = row.get("source_files_sha256")
        if not isinstance(hashes, dict):
            errors.append(f"{key(row)}: source hash map missing")
            continue
        for path in CORE_SOURCE_PATHS:
            value = hashes.get(path)
            if not isinstance(value, str):
                errors.append(f"{key(row)}: source hash missing for {path}")
            else:
                core_hashes[path].add(value)
    for path, values in core_hashes.items():
        if len(values) != 1:
            errors.append(f"core source mismatch for {path}: {sorted(values)}")

    pair_config_equal = True
    for dataset in DATASETS:
        for horizon in HORIZONS:
            for seed in SEEDS:
                for variant in VARIANTS:
                    timerole = observed[("timerole", dataset, horizon, seed, variant)]
                    recent = observed[("recent", dataset, horizon, seed, variant)]
                    timerole_config = dict(timerole.get("resolved_config") or {})
                    recent_config = dict(recent.get("resolved_config") or {})
                    for field in PAIR_CONFIG_EXCLUSIONS:
                        timerole_config.pop(field, None)
                        recent_config.pop(field, None)
                    if timerole_config != recent_config:
                        pair_config_equal = False
                        errors.append(
                            f"paired config mismatch: {dataset}/{horizon}/{seed}/{variant}"
                        )

    if errors:
        preview = "\n".join(errors[:40])
        suffix = "" if len(errors) <= 40 else f"\n... {len(errors) - 40} more"
        raise RuntimeError(f"Phase C integrity audit failed:\n{preview}{suffix}")
    return {
        "all_completed": True,
        "no_test_access": True,
        "all_source_clean": True,
        "validation_shuffle": False,
        "data_order_seed_matches_experiment_seed": True,
        "core_source_hashes": {
            path: next(iter(values)) for path, values in core_hashes.items()
        },
        "paired_configs_equal_except_identity": pair_config_equal,
        "source_commits": sorted(
            {str(row.get("git_commit_sha")) for row in observed.values()}
        ),
        "explicit_retry_records": sum(
            bool(row.get("retry_provenance")) for row in observed.values()
        ),
    }


def mean(rows: list[dict[str, object]], field: str) -> float:
    return statistics.mean(float(row[field]) for row in rows)


def pct_delta(value: float, baseline: float) -> float:
    return (value / baseline - 1.0) * 100.0


def reduction(candidate: list[dict[str, object]], baseline: list[dict[str, object]], field: str) -> float:
    return (1.0 - mean(candidate, field) / mean(baseline, field)) * 100.0


def rows_for(
    observed: dict[tuple[str, str, int, int, str], dict[str, object]],
    role: str,
    variant: str,
) -> list[dict[str, object]]:
    return [
        observed[(role, dataset, horizon, seed, variant)]
        for dataset in DATASETS
        for horizon in HORIZONS
        for seed in SEEDS
    ]


def analyze_variant(
    observed: dict[tuple[str, str, int, int, str], dict[str, object]],
    variant: str,
) -> dict[str, object]:
    baseline = rows_for(observed, "timerole", "R0")
    candidate = rows_for(observed, "timerole", variant)
    recent = rows_for(observed, "recent", variant)

    macro_mse = mean(candidate, "mse")
    macro_mae = mean(candidate, "mae")
    baseline_mse = mean(baseline, "mse")
    baseline_mae = mean(baseline, "mae")
    mse_nonlosses = sum(
        float(candidate_row["mse"]) <= float(baseline_row["mse"])
        for candidate_row, baseline_row in zip(candidate, baseline)
    )
    history_wins = sum(
        float(candidate_row["mse"]) < float(recent_row["mse"])
        for candidate_row, recent_row in zip(candidate, recent)
    )

    horizon_means: dict[str, object] = {}
    dataset_has_nonloss_horizon: dict[str, bool] = {}
    for dataset in DATASETS:
        nonloss_horizons = []
        for horizon in HORIZONS:
            candidate_cell = [
                observed[("timerole", dataset, horizon, seed, variant)]
                for seed in SEEDS
            ]
            baseline_cell = [
                observed[("timerole", dataset, horizon, seed, "R0")]
                for seed in SEEDS
            ]
            candidate_cell_mse = mean(candidate_cell, "mse")
            baseline_cell_mse = mean(baseline_cell, "mse")
            delta = pct_delta(candidate_cell_mse, baseline_cell_mse)
            horizon_means[f"{dataset}-{horizon}"] = {
                "candidate_mse_mean": candidate_cell_mse,
                "candidate_mse_sd": statistics.stdev(
                    float(row["mse"]) for row in candidate_cell
                ),
                "baseline_mse_mean": baseline_cell_mse,
                "mse_delta_pct": delta,
            }
            nonloss_horizons.append(delta <= 0.0)
        dataset_has_nonloss_horizon[dataset] = any(nonloss_horizons)

    resource_reductions = {
        field: reduction(candidate, baseline, field) for field in RESOURCE_KEYS
    }
    history_recent_mse = mean(recent, "mse")
    history_gain_pct = (
        (history_recent_mse - macro_mse) / history_recent_mse * 100.0
    )
    checks = {
        "mse_nonlosses_at_least_11_of_18": mse_nonlosses >= 11,
        "macro_mse_within_0_5pct": pct_delta(macro_mse, baseline_mse) <= 0.5,
        "macro_mae_within_0_5pct": pct_delta(macro_mae, baseline_mae) <= 0.5,
        "parameter_reduction_at_least_20pct": (
            resource_reductions["parameter_count"] >= 20.0
        ),
        "latency_reduction_at_least_15pct": (
            resource_reductions["milliseconds_per_batch"] >= 15.0
        ),
        "peak_memory_reduction_at_least_15pct": (
            resource_reductions["train_peak_cuda_memory_bytes"] >= 15.0
            or resource_reductions["peak_cuda_memory_bytes"] >= 15.0
        ),
        "every_dataset_has_nonloss_horizon": all(
            dataset_has_nonloss_horizon.values()
        ),
        "timerole_macro_mse_better_than_recent": history_gain_pct > 0.0,
        "timerole_history_wins_at_least_12_of_18": history_wins >= 12,
    }
    return {
        "variant": variant,
        "macro_mse": macro_mse,
        "macro_mse_delta_pct": pct_delta(macro_mse, baseline_mse),
        "macro_mae": macro_mae,
        "macro_mae_delta_pct": pct_delta(macro_mae, baseline_mae),
        "mse_nonlosses": mse_nonlosses,
        "history_recent_macro_mse": history_recent_mse,
        "history_gain_pct": history_gain_pct,
        "history_wins": history_wins,
        "resource_reduction_pct": resource_reductions,
        "dataset_has_nonloss_horizon": dataset_has_nonloss_horizon,
        "horizon_means": horizon_means,
        "checks": checks,
        "phase_c_pass": variant != "R0" and all(checks.values()),
    }


def write_ranking(results: dict[str, dict[str, object]]) -> None:
    fields = (
        "variant", "macro_mse", "macro_mse_delta_pct", "macro_mae",
        "macro_mae_delta_pct", "mse_nonlosses", "history_gain_pct",
        "history_wins", "parameter_reduction_pct", "latency_reduction_pct",
        "train_peak_memory_reduction_pct", "inference_peak_memory_reduction_pct",
        "phase_c_pass",
    )
    rows = []
    for variant in VARIANTS:
        result = results[variant]
        resources = result["resource_reduction_pct"]
        assert isinstance(resources, dict)
        rows.append({
            **{field: result.get(field) for field in fields},
            "parameter_reduction_pct": resources["parameter_count"],
            "latency_reduction_pct": resources["milliseconds_per_batch"],
            "train_peak_memory_reduction_pct": resources[
                "train_peak_cuda_memory_bytes"
            ],
            "inference_peak_memory_reduction_pct": resources[
                "peak_cuda_memory_bytes"
            ],
        })
    RANKING_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = RANKING_PATH.with_suffix(RANKING_PATH.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(RANKING_PATH)


def write_report(results: dict[str, dict[str, object]], passing: list[str]) -> None:
    lines = [
        "# TimeRole 近期预测器简化 Phase C 结果",
        "",
        f"更新时间：{now()}",
        "",
        "- 仅使用验证集；seed 2021 复用 Phase B，seeds 2022/2023 来自 Phase C。",
        "- 比较 R0、R2、R4 的 TimeRole 与严格匹配 Recent-only，共 108 个记录。",
        f"- 通过全部九项门槛的候选：{', '.join(passing) if passing else '无'}。",
        "",
        "| 候选 | 宏 MSE Δ | 宏 MAE Δ | R0 不劣胜场 | 历史收益 | 历史胜场 | 参数下降 | 延迟下降 | Phase C |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant in VARIANTS:
        result = results[variant]
        resources = result["resource_reduction_pct"]
        assert isinstance(resources, dict)
        lines.append(
            f"| {variant} | {float(result['macro_mse_delta_pct']):+.3f}% | "
            f"{float(result['macro_mae_delta_pct']):+.3f}% | "
            f"{int(result['mse_nonlosses'])}/18 | "
            f"{float(result['history_gain_pct']):+.3f}% | "
            f"{int(result['history_wins'])}/18 | "
            f"{float(resources['parameter_count']):+.1f}% | "
            f"{float(resources['milliseconds_per_batch']):+.1f}% | "
            f"{'PASS' if result['phase_c_pass'] else 'FAIL'} |"
        )
    lines.extend(["", "## 九项门槛", ""])
    for variant in ("R2", "R4"):
        checks = results[variant]["checks"]
        assert isinstance(checks, dict)
        lines.append(f"### {variant}")
        lines.append("")
        for name, passed in checks.items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
        lines.append("")
    atomic_text(REPORT_PATH, "\n".join(lines) + "\n")


def main() -> int:
    rows, metadata = selected_records()
    try:
        observed = validate_completeness(rows, metadata)
        integrity = validate_integrity(observed)
    except RuntimeError as error:
        print(str(error))
        return 2

    results = {
        variant: analyze_variant(observed, variant) for variant in VARIANTS
    }
    passing = [
        variant for variant in ("R2", "R4") if results[variant]["phase_c_pass"]
    ]
    payload = {
        "created_at": now(),
        "stage": "phase_c",
        "status": "analyzed",
        "validation_only": True,
        "test_accessed": False,
        "record_count": len(observed),
        "expected_record_count": 108,
        "reused_phase_b_seed": 2021,
        "new_phase_c_seeds": list(NEW_SEEDS),
        "roles": list(ROLES),
        "variants": list(VARIANTS),
        "datasets": list(DATASETS),
        "horizons": list(HORIZONS),
        "integrity": integrity,
        "gate_definition": {
            "minimum_mse_nonlosses": 11,
            "macro_mse_max_degradation_pct": 0.5,
            "macro_mae_max_degradation_pct": 0.5,
            "minimum_parameter_reduction_pct": 20.0,
            "minimum_latency_reduction_pct": 15.0,
            "minimum_peak_memory_reduction_pct": 15.0,
            "dataset_coverage": "at least one three-seed-mean horizon nonloss per dataset",
            "timerole_macro_mse_must_beat_recent": True,
            "minimum_timerole_history_wins": 12,
        },
        "passing_candidates": passing,
        "results": results,
        "ranking_csv": str(RANKING_PATH),
        "report_path": str(REPORT_PATH),
        **metadata,
    }
    atomic_json(GATE_PATH, payload)
    write_ranking(results)
    write_report(results, passing)
    print(json.dumps({
        "status": "analyzed",
        "record_count": len(observed),
        "passing_candidates": passing,
        "gate_json": str(GATE_PATH),
        "ranking_csv": str(RANKING_PATH),
        "report": str(REPORT_PATH),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
