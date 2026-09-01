#!/usr/bin/env python3
"""Plot test-set sensitivity of TimeRole's history-role boundary in Python."""

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
DEFAULT_HISTORY = ROOT / "logs/timerole_p0/history_length_test/summary.csv"
DEFAULT_BOUNDARY = ROOT / "logs/timerole_p0/boundary_pool_test/summary.csv"
DEFAULT_OUTPUT = ROOT / "paper/neurocomputing/figures/Fig4_History_Sensitivity_v1"
COLORS = {96: "#245B8A", 720: "#D17A45"}
MARKERS = {96: "o", 720: "s"}
DATASET_MARKERS = {"ETTm1": "o", "ETTm2": "s", "weather": "^"}


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
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def three_seed_means(path: Path, group_columns: list[str]) -> pd.DataFrame:
    raw = pd.read_csv(path)
    counts = raw.groupby(group_columns, dropna=False)["seed"].nunique()
    if not (counts == 3).all():
        bad = counts[counts != 3]
        raise ValueError(f"expected exactly three seeds in every cell:\n{bad}")
    return (
        raw.groupby(group_columns, as_index=False, dropna=False)
        .agg(test_mse=("test_mse", "mean"), test_mae=("test_mae", "mean"),
             parameter_count=("parameter_count", "mean"), seed_count=("seed", "nunique"))
    )


def prepare_source(history_path: Path, boundary_path: Path) -> pd.DataFrame:
    history = three_seed_means(history_path, ["dataset", "horizon", "seq_len"])
    standard = history[history["seq_len"] == 336][
        ["dataset", "horizon", "test_mse", "test_mae", "parameter_count"]
    ].rename(columns={"test_mse": "baseline_mse", "test_mae": "baseline_mae",
                      "parameter_count": "baseline_parameter_count"})
    if standard.groupby(["dataset", "horizon"]).size().ne(1).any():
        raise ValueError("standard L=336 configuration must be unique per task")

    history = history.merge(standard, on=["dataset", "horizon"], validate="many_to_one")
    history["panel"] = "total_history"
    history["setting"] = history["seq_len"].astype(int)

    boundary = three_seed_means(
        boundary_path, ["factor", "dataset", "horizon", "seq_len", "recent_len", "pool"]
    ).merge(standard, on=["dataset", "horizon"], validate="many_to_one")
    recent = boundary[boundary["factor"] == "recent"].copy()
    recent["panel"] = "recent_window"
    recent["setting"] = recent["recent_len"].astype(int)
    pooling = boundary[boundary["factor"] == "pool"].copy()
    pooling["panel"] = "pooling_width"
    pooling["setting"] = pooling["pool"].astype(int)

    # Add the standard setting explicitly to panels whose sweep files contain only alternatives.
    standard_boundary = standard[standard["dataset"].isin(boundary["dataset"].unique())].copy()
    standard_boundary["test_mse"] = standard_boundary["baseline_mse"]
    standard_boundary["test_mae"] = standard_boundary["baseline_mae"]
    standard_boundary["parameter_count"] = standard_boundary["baseline_parameter_count"]
    standard_boundary["seed_count"] = 3
    recent_standard = standard_boundary.copy()
    recent_standard["panel"], recent_standard["setting"] = "recent_window", 96
    pooling_standard = standard_boundary.copy()
    pooling_standard["panel"], pooling_standard["setting"] = "pooling_width", 16

    keep = ["panel", "dataset", "horizon", "setting", "test_mse", "test_mae",
            "parameter_count", "seed_count", "baseline_mse", "baseline_mae",
            "baseline_parameter_count"]
    source = pd.concat(
        [history[keep], recent[keep], recent_standard[keep], pooling[keep], pooling_standard[keep]],
        ignore_index=True,
    )
    source["mse_change_pct"] = 100.0 * (source["test_mse"] / source["baseline_mse"] - 1.0)
    source["mae_change_pct"] = 100.0 * (source["test_mae"] / source["baseline_mae"] - 1.0)
    source["parameter_change_pct"] = 100.0 * (
        source["parameter_count"] / source["baseline_parameter_count"] - 1.0
    )
    return source.sort_values(["panel", "horizon", "setting", "dataset"]).reset_index(drop=True)


def draw_panel(
    ax: plt.Axes,
    source: pd.DataFrame,
    panel: str,
    settings: list[int],
    title: str,
    xlabel: str,
) -> None:
    subset = source[source["panel"] == panel]
    positions = np.arange(len(settings), dtype=float)
    ax.axhline(0, color="#6E7377", lw=0.75, ls=(0, (3, 2)), zorder=1)
    ax.axvspan(positions[settings.index({"total_history": 336, "recent_window": 96,
                                       "pooling_width": 16}[panel])] - 0.33,
               positions[settings.index({"total_history": 336, "recent_window": 96,
                                         "pooling_width": 16}[panel])] + 0.33,
               color="#F1F3F4", zorder=-3)

    for horizon, offset in ((96, -0.07), (720, 0.07)):
        horizon_data = subset[subset["horizon"] == horizon]
        macro = horizon_data.groupby("setting")["mse_change_pct"].mean().reindex(settings)
        ax.plot(positions + offset, macro, color=COLORS[horizon], marker=MARKERS[horizon],
                ms=4.2, lw=1.35, markeredgecolor="white", markeredgewidth=0.45, zorder=4)
        for dataset, dataset_data in horizon_data.groupby("dataset"):
            values = dataset_data.set_index("setting")["mse_change_pct"].reindex(settings)
            ax.scatter(positions + offset, values, s=13, marker=DATASET_MARKERS[dataset],
                       facecolors=COLORS[horizon], edgecolors="white", linewidths=0.4,
                       alpha=0.38, zorder=3)

    ax.set_xticks(positions, [str(value) for value in settings])
    ax.set_xlim(-0.45, len(settings) - 0.55)
    ax.set_title(title, loc="left", pad=6, fontweight="bold")
    ax.set_xlabel(xlabel, labelpad=3)
    ax.yaxis.grid(True, color="#E5E7E9", lw=0.45, zorder=-4)
    ax.tick_params(length=2.4, color="#777777")
    ax.spines["bottom"].set_color("#666666")
    ax.spines["left"].set_color("#666666")
    ax.text(positions[settings.index({"total_history": 336, "recent_window": 96,
                                     "pooling_width": 16}[panel])], 5.85, "standard",
            ha="center", va="top", fontsize=5.5, color="#73777A")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--boundary", type=Path, default=DEFAULT_BOUNDARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="output base path without an extension")
    args = parser.parse_args()
    paths = [p if p.is_absolute() else ROOT / p for p in (args.history, args.boundary)]
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    configure_style()
    source = prepare_source(*paths)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharey=True, facecolor="white")
    panels = (
        ("total_history", [192, 336, 720, 960], "a  Total history", "History length $L$"),
        ("recent_window", [48, 96, 192], "b  Recent high-resolution window",
         "Recent length $L_r$"),
        ("pooling_width", [8, 16, 24], "c  Historical pooling width", "Pooling width $r$"),
    )
    for ax, spec in zip(axes, panels):
        draw_panel(ax, source, *spec)
    axes[0].set_ylabel("Test MSE change vs standard (%)")
    axes[0].set_ylim(-5.2, 6.2)

    horizon_handles = [
        Line2D([0], [0], color=COLORS[h], marker=MARKERS[h], lw=1.35, ms=4.2,
               markeredgecolor="white", markeredgewidth=0.45, label=f"H = {h}")
        for h in (96, 720)
    ]
    dataset_handles = [
        Line2D([0], [0], color="#777777", marker=DATASET_MARKERS[d], lw=0, ms=3.8,
               markerfacecolor="#777777", markeredgecolor="white", markeredgewidth=0.4,
               label=d)
        for d in ("ETTm1", "ETTm2", "weather")
    ]
    dataset_handles[-1].set_label("Weather")
    fig.legend(handles=horizon_handles + dataset_handles, loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.01), handlelength=1.6, columnspacing=1.3)
    fig.text(0.995, 0.017, "Small markers: dataset means (3 seeds); lines: macro means",
             ha="right", va="bottom", fontsize=5.4, color="#696D70")
    fig.subplots_adjust(left=0.078, right=0.995, bottom=0.23, top=0.80, wspace=0.19)

    source.to_csv(output.with_name(output.name + "_source_data.csv"), index=False)
    for extension in ("svg", "pdf", "png", "tiff"):
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if extension in {"png", "tiff"}:
            kwargs["dpi"] = 600
        fig.savefig(output.with_suffix(f".{extension}"), **kwargs)
    plt.close(fig)

    macro = source.groupby(["panel", "horizon", "setting"], as_index=False)[
        ["mse_change_pct", "mae_change_pct", "parameter_change_pct"]
    ].mean()
    macro.to_csv(output.with_name(output.name + "_macro_summary.csv"), index=False)
    print(f"wrote {output}.{{svg,pdf,png,tiff}}")
    print(f"source_rows={len(source)} macro_rows={len(macro)}")


if __name__ == "__main__":
    main()
