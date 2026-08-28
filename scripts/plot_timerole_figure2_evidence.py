#!/usr/bin/env python3
"""Generate the TimeRole three-panel evidence figure from experiment CSV files."""

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
CORE_RESULTS = ROOT / "logs/timerole_core_ablation/results_long.csv"
INTERVENTIONS = ROOT / "logs/cmrhm_interventions/summary.csv"
TRANSFER = ROOT / "logs/timexer336_dhc_multiseed/paired_test_results.csv"
DEFAULT_OUTPUT = ROOT / "paper/neurocomputing/figures/Fig2_TimeRole_Evidence_v1"

DATASET_COLORS = {"ETTm1": "#245B8A", "ETTm2": "#D17A45"}
HORIZON_MARKERS = {96: "o", 192: "s", 336: "D", 720: "^"}
INTERVENTION_ORDER = [
    "intact",
    "batch_shuffle",
    "temporal_shuffle",
    "reverse",
    "recent_mean",
    "noise",
]
INTERVENTION_LABELS = ["Intact", "Mismatch", "Shuffle", "Reverse", "Mean", "Noise"]
TASK_ORDER = [("ETTm1", 96), ("ETTm1", 720), ("ETTm2", 96), ("ETTm2", 720)]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "axes.linewidth": 0.75,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 5.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_primary_pairs() -> pd.DataFrame:
    raw = pd.read_csv(CORE_RESULTS)
    raw = raw[
        raw["dataset"].isin(DATASET_COLORS)
        & raw["variant"].isin(["full", "no_dhc"])
    ].copy()
    wide = raw.pivot(
        index=["dataset", "horizon", "seed"],
        columns="variant",
        values=["test_mse", "test_mae"],
    )
    out = wide.index.to_frame(index=False)
    for metric in ("mse", "mae"):
        full = wide[(f"test_{metric}", "full")].to_numpy()
        baseline = wide[(f"test_{metric}", "no_dhc")].to_numpy()
        out[f"{metric}_improvement_pct"] = 100.0 * (baseline - full) / baseline
    out["panel"] = "a"
    out["comparison"] = "TimeRole vs RGSP-96"
    out["split"] = "test"
    return out


def load_intervention_matrix() -> tuple[pd.DataFrame, np.ndarray]:
    raw = pd.read_csv(INTERVENTIONS)
    raw = raw[
        raw["intervention"].isin(INTERVENTION_ORDER)
        & raw[["dataset", "horizon"]].apply(tuple, axis=1).isin(TASK_ORDER)
    ].copy()
    lookup = raw.set_index(["dataset", "horizon", "intervention"])["mse_change_pct"]
    matrix = np.array(
        [[lookup.loc[dataset, horizon, intervention] for intervention in INTERVENTION_ORDER]
         for dataset, horizon in TASK_ORDER],
        dtype=float,
    )
    raw["panel"] = "b"
    raw["comparison"] = "Intervention vs intact history"
    raw["split"] = "validation"
    raw["seed"] = 2021
    return raw, matrix


def load_transfer_pairs() -> pd.DataFrame:
    raw = pd.read_csv(TRANSFER).rename(columns={"pred_len": "horizon"})
    raw["panel"] = "c"
    raw["comparison"] = "TimeXer+DHC vs TimeXer-336"
    raw["split"] = "test"
    return raw


def validate_data(primary: pd.DataFrame, intervention_matrix: np.ndarray, transfer: pd.DataFrame) -> None:
    if len(primary) != 24:
        raise ValueError(f"Panel a requires 24 paired seed results, found {len(primary)}")
    if not ((primary["mse_improvement_pct"] > 0).all() and (primary["mae_improvement_pct"] > 0).all()):
        raise ValueError("Panel a contains a non-positive paired improvement")
    if intervention_matrix.shape != (4, 6):
        raise ValueError(f"Panel b requires a 4x6 matrix, found {intervention_matrix.shape}")
    if not np.allclose(intervention_matrix[:, 0], 0.0):
        raise ValueError("Panel b intact-history column must be zero")
    if not (intervention_matrix[:, 1:] > 0).all():
        raise ValueError("Panel b contains a non-positive intervention effect")
    if not np.all(np.argmax(intervention_matrix[:, 1:], axis=1) == 0):
        raise ValueError("Mismatch is not the strongest intervention in every task")
    if len(transfer) != 12:
        raise ValueError(f"Panel c requires 12 paired seed results, found {len(transfer)}")
    mse_wins = int((transfer["mse_improvement_pct"] > 0).sum())
    mae_wins = int((transfer["mae_improvement_pct"] > 0).sum())
    if (mse_wins, mae_wins) != (10, 8):
        raise ValueError(f"Unexpected transfer win counts: MSE={mse_wins}, MAE={mae_wins}")


def style_improvement_axis(
    ax: plt.Axes,
    title: str,
    note: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xticks: list[float],
    yticks: list[float],
) -> None:
    ax.axhline(0, color="#A8A8A8", lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.axvline(0, color="#A8A8A8", lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlabel("MSE improvement (%)")
    ax.set_ylabel("MAE improvement (%)")
    ax.set_title(title, loc="left", pad=18, fontweight="bold")
    ax.text(0.0, 1.015, note, transform=ax.transAxes, fontsize=5.6,
            color="#555555", va="bottom")


def draw_pair_points(ax: plt.Axes, data: pd.DataFrame) -> None:
    for (dataset, horizon), group in data.groupby(["dataset", "horizon"], sort=False):
        ax.scatter(
            group["mse_improvement_pct"],
            group["mae_improvement_pct"],
            s=28,
            marker=HORIZON_MARKERS[int(horizon)],
            facecolor=DATASET_COLORS[dataset],
            edgecolor="white",
            linewidth=0.55,
            alpha=0.82,
            zorder=3,
        )


def draw_panel_a(ax: plt.Axes, data: pd.DataFrame) -> None:
    draw_pair_points(ax, data)
    style_improvement_axis(
        ax,
        "DHC consistently improves the RGSP backbone",
        "Test set · individual paired seeds (n=24) · all improve both metrics",
        xlim=(-1.5, 22.5),
        ylim=(-0.6, 8.3),
        xticks=[0, 5, 10, 15, 20],
        yticks=[0, 2, 4, 6, 8],
    )
    handles = [
        Line2D([0], [0], marker="o", lw=0, markerfacecolor=color, markeredgecolor="white",
               markersize=5.5, label=dataset)
        for dataset, color in DATASET_COLORS.items()
    ]
    handles.extend(
        Line2D([0], [0], marker=marker, lw=0, markerfacecolor="#737373", markeredgecolor="white",
               markersize=5.2, label=f"H={horizon}")
        for horizon, marker in HORIZON_MARKERS.items()
    )
    ax.legend(handles=handles, loc="upper left", ncol=2, columnspacing=0.8,
              handletextpad=0.35, borderaxespad=0.5)


def draw_panel_b(ax: plt.Axes, matrix: np.ndarray) -> None:
    norm = mpl.colors.Normalize(vmin=0.0, vmax=80.0)
    cmap = mpl.colormaps["OrRd"]
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            rgba = cmap(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "white" if luminance < 0.53 else "#252525"
            ax.text(col, row, f"{value:.1f}", ha="center", va="center",
                    fontsize=5.6, color=text_color, fontweight="bold" if col == 1 else "normal")
    ax.set_xticks(np.arange(len(INTERVENTION_LABELS)), INTERVENTION_LABELS, rotation=38, ha="right")
    ax.set_yticks(np.arange(len(TASK_ORDER)), [f"{d}–{h}" for d, h in TASK_ORDER])
    ax.tick_params(length=0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(0.5, color="white", lw=1.3)
    ax.set_title("History interventions raise error", loc="left", pad=18, fontweight="bold")
    ax.text(0.0, 1.015, "Δ validation MSE (%) · frozen checkpoint, seed 2021",
            transform=ax.transAxes, fontsize=5.5, color="#555555", va="bottom")


def draw_panel_c(ax: plt.Axes, data: pd.DataFrame) -> None:
    draw_pair_points(ax, data)
    style_improvement_axis(
        ax,
        "DHC transfers across backbones",
        "Test · paired seeds (n=12) · wins: MSE 10, MAE 8",
        xlim=(-2.5, 6.5),
        ylim=(-1.0, 3.5),
        xticks=[-2, 0, 2, 4, 6],
        yticks=[-1, 0, 1, 2, 3],
    )


def write_source_data(primary: pd.DataFrame, intervention: pd.DataFrame,
                      transfer: pd.DataFrame, output_base: Path) -> None:
    columns = [
        "panel", "comparison", "split", "dataset", "horizon", "seed",
        "intervention", "mse_improvement_pct", "mae_improvement_pct",
    ]
    a = primary.copy()
    a["intervention"] = ""
    b = intervention.rename(
        columns={"mse_change_pct": "mse_improvement_pct", "mae_change_pct": "mae_improvement_pct"}
    ).copy()
    c = transfer.copy()
    c["intervention"] = ""
    source = pd.concat(
        [a.reindex(columns=columns), b.reindex(columns=columns), c.reindex(columns=columns)],
        ignore_index=True,
    )
    source.to_csv(output_base.with_name(output_base.name + "_source_data.csv"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output path without an extension")
    args = parser.parse_args()
    output_base = args.output if args.output.is_absolute() else ROOT / args.output
    output_base.parent.mkdir(parents=True, exist_ok=True)

    configure_style()
    primary = load_primary_pairs()
    intervention, intervention_matrix = load_intervention_matrix()
    transfer = load_transfer_pairs()
    validate_data(primary, intervention_matrix, transfer)

    fig = plt.figure(figsize=(7.2, 4.45), facecolor="white")
    grid = fig.add_gridspec(
        2,
        5,
        width_ratios=[1.14, 1.14, 1.14, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.075,
        right=0.985,
        bottom=0.13,
        top=0.91,
        wspace=1.05,
        hspace=0.88,
    )
    ax_a = fig.add_subplot(grid[:, :3])
    ax_b = fig.add_subplot(grid[0, 3:])
    ax_c = fig.add_subplot(grid[1, 3:])

    draw_panel_a(ax_a, primary)
    draw_panel_b(ax_b, intervention_matrix)
    draw_panel_c(ax_c, transfer)

    for label, ax in zip("abc", (ax_a, ax_b, ax_c)):
        ax.text(-0.12 if label == "a" else -0.15, 1.08, label, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top", ha="left")

    svg_metadata = {
        "Title": "TimeRole evidence figure",
        "Description": "DHC effectiveness, distant-history interventions, and backbone transfer",
        "Creator": "Python matplotlib",
    }
    pdf_metadata = {
        "Title": "TimeRole evidence figure",
        "Subject": "DHC effectiveness, distant-history interventions, and backbone transfer",
        "Creator": "Python matplotlib",
    }
    fig.savefig(output_base.with_suffix(".svg"), metadata=svg_metadata)
    fig.savefig(output_base.with_suffix(".pdf"), metadata=pdf_metadata)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, metadata=svg_metadata)
    fig.savefig(
        output_base.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    write_source_data(primary, intervention, transfer, output_base)

    print(f"panel_a_rows={len(primary)}")
    print(f"panel_a_mse_range={primary['mse_improvement_pct'].min():.3f}..{primary['mse_improvement_pct'].max():.3f}")
    print(f"panel_a_mae_range={primary['mae_improvement_pct'].min():.3f}..{primary['mae_improvement_pct'].max():.3f}")
    print(f"panel_b_shape={intervention_matrix.shape[0]}x{intervention_matrix.shape[1]}")
    print(f"panel_b_max={intervention_matrix[:, 1:].max():.3f}")
    print(f"panel_c_rows={len(transfer)}")
    print(f"outputs={output_base}")


if __name__ == "__main__":
    main()
