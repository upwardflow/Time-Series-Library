#!/usr/bin/env python3
"""Plot TimeRole accuracy–efficiency trade-offs from measured validation runs."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "timerole-matplotlib"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "logs/cmrhm_efficiency/summary.csv"
DEFAULT_OUTPUT = ROOT / "paper/neurocomputing/figures/Fig5_Efficiency_Tradeoff_v1"
METHODS = ("RGSP-96", "TimeRole", "RGSP-336")
VARIANT_TO_METHOD = {"Recent96": "RGSP-96", "CMRHM": "TimeRole", "Raw336": "RGSP-336"}
COLORS = {"RGSP-96": "#8C9296", "TimeRole": "#245B8A", "RGSP-336": "#D17A45"}
MARKERS = {"RGSP-96": "o", "TimeRole": "o", "RGSP-336": "s"}


def configure_style() -> None:
    mpl.rcParams.update({
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
        "legend.fontsize": 6.1,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def prepare(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {
        "dataset", "horizon", "variant", "validation_mse", "parameter_count",
        "milliseconds_per_batch", "peak_cuda_memory_bytes",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"missing efficiency columns: {sorted(missing)}")
    if set(raw["dataset"]) != {"ETTm1", "ETTm2"}:
        raise ValueError("efficiency figure requires matched ETTm1/ETTm2 measurements")
    if set(raw["horizon"]) != {96, 720} or len(raw) != 12:
        raise ValueError("efficiency figure requires 12 complete cells")
    raw = raw.assign(method=raw["variant"].map(VARIANT_TO_METHOD))
    if raw["method"].isna().any():
        raise ValueError("unrecognized efficiency variant")
    summary = raw.groupby(["horizon", "method"], as_index=False).agg(
        validation_mse=("validation_mse", "mean"),
        parameter_count=("parameter_count", "mean"),
        milliseconds_per_batch=("milliseconds_per_batch", "mean"),
        peak_cuda_memory_bytes=("peak_cuda_memory_bytes", "mean"),
        dataset_count=("dataset", "nunique"),
    )
    if len(summary) != 6 or not (summary["dataset_count"] == 2).all():
        raise ValueError("incomplete horizon–method aggregation")
    summary["parameters_million"] = summary["parameter_count"] / 1e6
    summary["peak_memory_mib"] = summary["peak_cuda_memory_bytes"] / 1024**2
    return summary


def draw_panel(ax: plt.Axes, data: pd.DataFrame, horizon: int, label: str) -> None:
    panel = data[data["horizon"] == horizon].set_index("method").loc[list(METHODS)].reset_index()
    for _, row in panel.iterrows():
        method = row["method"]
        ax.scatter(
            row["milliseconds_per_batch"], row["validation_mse"],
            s=row["peak_memory_mib"] * 1.35,
            marker=MARKERS[method], facecolor=COLORS[method], edgecolor="white",
            linewidth=0.9, alpha=0.88, zorder=4,
        )

    offsets = {
        96: {"RGSP-96": (-2, 10), "TimeRole": (10, -7), "RGSP-336": (-53, 11)},
        720: {"RGSP-96": (-2, 10), "TimeRole": (10, -7), "RGSP-336": (-53, 11)},
    }
    for _, row in panel.iterrows():
        method = row["method"]
        text = f"{method}\n{row['parameters_million']:.2f}M · {row['peak_memory_mib']:.0f} MiB"
        ax.annotate(
            text,
            (row["milliseconds_per_batch"], row["validation_mse"]),
            xytext=offsets[horizon][method], textcoords="offset points",
            ha="left", va="center", fontsize=5.7, color=COLORS[method],
            fontweight="bold" if method == "TimeRole" else "normal",
            zorder=5,
        )

    ax.set_title(f"{label}  Forecast horizon H = {horizon}", loc="left", pad=6,
                 fontweight="bold")
    ax.set_xlabel("Latency (ms/batch, ↓)")
    ax.set_ylabel("Validation MSE (↓)")
    ax.set_xlim(4.0, 9.8)
    y_min = panel["validation_mse"].min()
    y_max = panel["validation_mse"].max()
    pad = 0.30 * (y_max - y_min)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.grid(True, color="#E4E7E9", lw=0.45, zorder=-3)
    ax.tick_params(length=2.4, color="#777777")
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    configure_style()
    source = prepare(input_path)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), facecolor="white")
    draw_panel(axes[0], source, 96, "a")
    draw_panel(axes[1], source, 720, "b")

    size_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#B7BCC0",
               markeredgecolor="white", markersize=(memory * 0.27) ** 0.5,
               label=f"{memory} MiB")
        for memory in (150, 300, 450)
    ]
    fig.legend(handles=size_handles, title="Bubble area ∝ peak memory", loc="upper center",
               bbox_to_anchor=(0.5, 1.005), ncol=3, handletextpad=0.5,
               columnspacing=1.2, title_fontsize=6.1)
    fig.subplots_adjust(left=0.082, right=0.992, bottom=0.20, top=0.79, wspace=0.27)

    metadata = {
        "Title": "TimeRole accuracy-efficiency trade-off",
        "Description": "Validation MSE, batch latency, peak memory, and parameter count",
        "Creator": "Python matplotlib",
    }
    fig.savefig(output.with_suffix(".svg"), metadata=metadata)
    fig.savefig(output.with_suffix(".pdf"), metadata={
        "Title": metadata["Title"], "Subject": metadata["Description"],
        "Creator": metadata["Creator"],
    })
    fig.savefig(output.with_suffix(".png"), dpi=600, metadata=metadata)
    fig.savefig(output.with_suffix(".tiff"), dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    source.to_csv(output.with_name(output.name + "_source_data.csv"), index=False)
    print(f"source_rows={len(source)}")
    print(f"outputs={output}")


if __name__ == "__main__":
    main()
