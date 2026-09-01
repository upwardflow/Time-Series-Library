#!/usr/bin/env python3
"""Plot publication-ready real forecast profiles from compact checkpoint exports."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "timerole-matplotlib"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "paper/neurocomputing/figures/forecast_profiles/source_npz"
DEFAULT_OUTPUT = ROOT / "paper/neurocomputing/figures/Fig3_Forecast_Profiles_v1"
DATASET_ROWS = (
    ("ETTm1", "RGSP-96", "15-min"),
    ("ETTh1", "DLinear", "hourly"),
)
HORIZONS = (96, 720)
COLORS = {
    "context": "#A6ABB0",
    "truth": "#202124",
    "TimeRole": "#245B8A",
    "baseline": "#D17A45",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "legend.fontsize": 6,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_profile(input_dir: Path, dataset: str, horizon: int, label: str) -> dict[str, object]:
    slug = label.lower().replace("-", "_")
    path = input_dir / f"{dataset.lower()}_h{horizon}_{slug}.npz"
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def validate_profiles(profiles: dict[tuple[str, int, str], dict[str, object]]) -> None:
    for dataset, baseline, _ in DATASET_ROWS:
        for horizon in HORIZONS:
            ours = profiles[(dataset, horizon, "TimeRole")]
            other = profiles[(dataset, horizon, baseline)]
            if int(ours["origin"]) != int(other["origin"]):
                raise ValueError(f"origin mismatch for {dataset} H={horizon}")
            for key in ("context", "target"):
                if not np.allclose(ours[key], other[key], rtol=1e-6, atol=1e-5):
                    raise ValueError(f"{key} mismatch for {dataset} H={horizon}")
            if len(ours["context"]) != 96 or len(ours["target"]) != horizon:
                raise ValueError(f"unexpected profile shape for {dataset} H={horizon}")
        if not np.allclose(
            profiles[(dataset, 96, "TimeRole")]["target"],
            profiles[(dataset, 720, "TimeRole")]["target"][:96],
            rtol=1e-6,
            atol=1e-5,
        ):
            raise ValueError(f"short-horizon truth is not the long-horizon prefix for {dataset}")


def window_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    error = prediction - target
    return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error * error)))


def row_limits(
    profiles: dict[tuple[str, int, str], dict[str, object]], dataset: str, baseline: str
) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for horizon in HORIZONS:
        ours = profiles[(dataset, horizon, "TimeRole")]
        other = profiles[(dataset, horizon, baseline)]
        values.extend([ours["context"], ours["target"], ours["prediction"], other["prediction"]])
    combined = np.concatenate(values)
    span = float(combined.max() - combined.min())
    padding = max(0.04 * span, 0.05)
    return float(combined.min() - padding), float(combined.max() + padding)


def draw_panel(
    ax: plt.Axes,
    ours: dict[str, object],
    baseline: dict[str, object],
    baseline_label: str,
    horizon: int,
    ylim: tuple[float, float],
) -> tuple[dict[str, float | str | int], dict[str, float | str | int]]:
    context = np.asarray(ours["context"], dtype=float)
    truth = np.asarray(ours["target"], dtype=float)
    ours_pred = np.asarray(ours["prediction"], dtype=float)
    baseline_pred = np.asarray(baseline["prediction"], dtype=float)
    context_x = np.arange(-len(context), 0)
    forecast_x = np.arange(horizon)

    ax.axvspan(-len(context), 0, color="#F1F3F4", zorder=-4)
    ax.axvline(0, color="#7E8387", lw=0.75, ls=(0, (3, 2)), zorder=1)
    ax.plot(context_x, context, color=COLORS["context"], lw=0.85, zorder=2)
    ax.plot(forecast_x, truth, color=COLORS["truth"], lw=0.95, zorder=4)
    ax.plot(forecast_x, baseline_pred, color=COLORS["baseline"], lw=0.8, alpha=0.92, zorder=2)
    ax.plot(forecast_x, ours_pred, color=COLORS["TimeRole"], lw=1.0, alpha=0.96, zorder=3)

    ours_mae, ours_rmse = window_metrics(ours_pred, truth)
    base_mae, base_rmse = window_metrics(baseline_pred, truth)
    ax.text(
        0.985,
        0.965,
        f"window MAE  {ours_mae:.2f} / {base_mae:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.4,
        color="#55585B",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 1.0, "pad": 1.2},
        zorder=6,
    )
    ax.text(
        0.985,
        0.89,
        f"TimeRole / {baseline_label}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.1,
        color="#777A7D",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 1.0, "pad": 1.0},
        zorder=6,
    )

    ax.set_xlim(-96, horizon - 1)
    ax.set_ylim(*ylim)
    if horizon == 96:
        ax.set_xticks([-96, -48, 0, 48, 95], ["−96", "−48", "0", "48", "96"])
    else:
        ax.set_xticks([-96, 0, 240, 480, 719], ["−96", "0", "240", "480", "720"])
    ax.yaxis.grid(True, color="#E5E7E9", lw=0.45, zorder=-3)
    ax.tick_params(length=2.5, color="#777777")
    ax.spines["bottom"].set_color("#666666")
    ax.spines["left"].set_color("#666666")

    common = {
        "dataset": str(ours["dataset"]),
        "horizon": horizon,
        "origin": int(ours["origin"]),
    }
    return (
        {**common, "model": "TimeRole", "mae": ours_mae, "rmse": ours_rmse},
        {**common, "model": baseline_label, "mae": base_mae, "rmse": base_rmse},
    )


def source_rows(
    profiles: dict[tuple[str, int, str], dict[str, object]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for dataset, baseline_label, _ in DATASET_ROWS:
        for horizon in HORIZONS:
            ours = profiles[(dataset, horizon, "TimeRole")]
            baseline = profiles[(dataset, horizon, baseline_label)]
            origin = int(ours["origin"])
            for step, value in zip(range(-96, 0), ours["context"]):
                rows.append({"dataset": dataset, "horizon": horizon, "origin": origin,
                             "phase": "observed_context", "step": step,
                             "series": "Observed history", "value": float(value)})
            for step, value in enumerate(ours["target"]):
                rows.append({"dataset": dataset, "horizon": horizon, "origin": origin,
                             "phase": "forecast", "step": step,
                             "series": "Ground truth", "value": float(value)})
            for label, profile in (("TimeRole", ours), (baseline_label, baseline)):
                prediction = np.asarray(profile["prediction"], dtype=float)
                target = np.asarray(profile["target"], dtype=float)
                for step, value in enumerate(prediction):
                    rows.append({"dataset": dataset, "horizon": horizon, "origin": origin,
                                 "phase": "forecast", "step": step,
                                 "series": label, "value": float(value)})
                mae, rmse = window_metrics(prediction, target)
                summaries.append({"dataset": dataset, "horizon": horizon, "origin": origin,
                                  "model": label, "window_mae_original_scale": mae,
                                  "window_rmse_original_scale": rmse})
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="output base path without an extension")
    args = parser.parse_args()
    input_dir = args.input if args.input.is_absolute() else ROOT / args.input
    output_base = args.output if args.output.is_absolute() else ROOT / args.output
    output_base.parent.mkdir(parents=True, exist_ok=True)

    configure_style()
    profiles: dict[tuple[str, int, str], dict[str, object]] = {}
    for dataset, baseline, _ in DATASET_ROWS:
        for horizon in HORIZONS:
            for label in ("TimeRole", baseline):
                profiles[(dataset, horizon, label)] = load_profile(
                    input_dir, dataset, horizon, label
                )
    validate_profiles(profiles)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.35), sharex=False, facecolor="white")
    summary_rows: list[dict[str, float | str | int]] = []
    panel_labels = iter("abcd")
    for row, (dataset, baseline_label, frequency) in enumerate(DATASET_ROWS):
        ylim = row_limits(profiles, dataset, baseline_label)
        for col, horizon in enumerate(HORIZONS):
            ax = axes[row, col]
            summaries = draw_panel(
                ax,
                profiles[(dataset, horizon, "TimeRole")],
                profiles[(dataset, horizon, baseline_label)],
                baseline_label,
                horizon,
                ylim,
            )
            summary_rows.extend(summaries)
            if row == 0:
                ax.set_title(f"Forecast horizon H = {horizon}", fontweight="bold", pad=7)
            if col == 0:
                ax.set_ylabel(f"{dataset} · OT\n({frequency})", labelpad=7)
            if row == 1:
                ax.set_xlabel("Time relative to forecast start (steps)", labelpad=4)
            ax.text(-0.105, 1.055, next(panel_labels), transform=ax.transAxes,
                    fontsize=9, fontweight="bold", va="top", ha="left")

    legend = [
        Line2D([0], [0], color=COLORS["context"], lw=1.2, label="Observed history"),
        Line2D([0], [0], color=COLORS["truth"], lw=1.2, label="Ground truth"),
        Line2D([0], [0], color=COLORS["TimeRole"], lw=1.4, label="TimeRole"),
        Line2D([0], [0], color=COLORS["baseline"], lw=1.2, label="Comparison model"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.56, 0.995),
               ncol=4, columnspacing=1.15, handlelength=2.2, handletextpad=0.45)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.13, top=0.89,
                        wspace=0.22, hspace=0.34)

    metadata = {
        "Title": "Real forecast profiles across prediction horizons",
        "Description": "Deterministically selected ETTm1 and ETTh1 OT forecasts from frozen checkpoints",
        "Creator": "Python matplotlib",
    }
    fig.savefig(output_base.with_suffix(".svg"), metadata=metadata)
    fig.savefig(output_base.with_suffix(".pdf"), metadata={
        "Title": metadata["Title"], "Subject": metadata["Description"],
        "Creator": metadata["Creator"],
    })
    fig.savefig(output_base.with_suffix(".png"), dpi=600, metadata=metadata)
    fig.savefig(output_base.with_suffix(".tiff"), dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    source, summary = source_rows(profiles)
    source.to_csv(output_base.with_name(output_base.name + "_source_data.csv"), index=False)
    summary.to_csv(output_base.with_name(output_base.name + "_window_metrics.csv"), index=False)
    print(f"source_rows={len(source)}")
    print(f"summary_rows={len(summary)}")
    print(f"outputs={output_base}")


if __name__ == "__main__":
    main()
