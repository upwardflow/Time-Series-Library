#!/usr/bin/env python3
"""Generate the reference-informed TimeRole Figure 2 v2 with Python/matplotlib.

The v2 keeps the validated data-loading and integrity checks from the v1 script,
but replaces the legend-heavy two-dimensional scatters with aligned effect rows.
"""

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
from matplotlib.patches import Rectangle

from plot_timerole_figure2_evidence import (
    INTERVENTION_LABELS,
    TASK_ORDER,
    load_intervention_matrix,
    load_primary_pairs,
    load_transfer_pairs,
    validate_data,
    write_source_data,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper/neurocomputing/figures/Fig2_TimeRole_Evidence_v2"

METRIC_STYLE = {
    "mse": {"label": "MSE", "color": "#245B8A", "marker": "o", "offset": 0.15},
    "mae": {"label": "MAE", "color": "#D17A45", "marker": "s", "offset": -0.15},
}
NEUTRAL = "#777777"
GRID = "#E3E5E7"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
            "font.size": 7.4,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.4,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def ordered_groups(data: pd.DataFrame, order: list[tuple[str, int]]) -> list[tuple[tuple[str, int], pd.DataFrame]]:
    groups: list[tuple[tuple[str, int], pd.DataFrame]] = []
    indexed = data.set_index(["dataset", "horizon"])
    for dataset, horizon in order:
        group = indexed.loc[(dataset, horizon)].reset_index()
        groups.append(((dataset, horizon), group))
    return groups


def add_dataset_bands(ax: plt.Axes, n_rows: int, split_after: int) -> None:
    ax.axhspan(n_rows - split_after - 0.5, n_rows - 0.5, color="#F5F7F8", zorder=-4)
    ax.axhline(n_rows - split_after - 0.5, color="#BFC4C8", lw=0.65, zorder=-1)


def draw_effect_rows(
    ax: plt.Axes,
    data: pd.DataFrame,
    task_order: list[tuple[str, int]],
    *,
    title: str,
    note: str,
    xlim: tuple[float, float],
    xticks: list[float],
    xlabel: str,
    show_legend: bool,
) -> None:
    groups = ordered_groups(data, task_order)
    n_rows = len(groups)
    y_base = np.arange(n_rows - 1, -1, -1, dtype=float)

    add_dataset_bands(ax, n_rows=n_rows, split_after=sum(d == "ETTm1" for d, _ in task_order))
    ax.axvline(0, color="#969A9D", lw=0.75, ls=(0, (3, 2)), zorder=-1)
    ax.xaxis.grid(True, color=GRID, lw=0.5, zorder=-3)

    for row, ((_, _), group) in zip(y_base, groups):
        for metric, style in METRIC_STYLE.items():
            values = group[f"{metric}_improvement_pct"].to_numpy(dtype=float)
            y = row + style["offset"]
            ax.plot(
                [values.min(), values.max()],
                [y, y],
                color=style["color"],
                lw=1.05,
                alpha=0.82,
                solid_capstyle="round",
                zorder=2,
            )
            ax.scatter(
                values,
                np.full_like(values, y),
                s=12,
                marker=style["marker"],
                facecolor=style["color"],
                edgecolor="white",
                linewidth=0.4,
                alpha=0.58,
                zorder=3,
            )
            ax.scatter(
                [values.mean()],
                [y],
                s=31,
                marker=style["marker"],
                facecolor=style["color"],
                edgecolor="white",
                linewidth=0.65,
                zorder=4,
            )

    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_ylim(-0.65, n_rows - 0.35)
    ax.set_yticks(y_base, [f"{dataset}  {horizon}" for dataset, horizon in task_order])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", length=2.5, color=NEUTRAL)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#555555")
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_title(title, loc="left", pad=17, fontweight="bold")
    ax.text(0.0, 1.015, note, transform=ax.transAxes, fontsize=6.0, color="#555555", va="bottom")

    if show_legend:
        handles = [
            Line2D(
                [0], [0], marker=style["marker"], lw=1.0, color=style["color"],
                markerfacecolor=style["color"], markeredgecolor="white", markersize=5.0,
                label=style["label"],
            )
            for style in METRIC_STYLE.values()
        ]
        ax.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.105),
            ncol=2,
            handlelength=1.45,
            columnspacing=0.9,
            handletextpad=0.4,
            borderaxespad=0,
        )


def draw_panel_b(ax: plt.Axes, matrix: np.ndarray) -> None:
    norm = mpl.colors.Normalize(vmin=0.0, vmax=80.0)
    cmap = mpl.colormaps["OrRd"]
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            rgba = cmap(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(
                col,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=6.0,
                color="white" if luminance < 0.53 else "#252525",
                fontweight="bold" if col == 1 else "normal",
            )

    # The outline directs attention to the content-mismatch intervention without
    # changing the quantitative color encoding.
    ax.add_patch(Rectangle((0.5, -0.5), 1.0, matrix.shape[0], fill=False,
                           edgecolor="#7F0000", linewidth=1.15, clip_on=False))
    ax.set_xticks(np.arange(len(INTERVENTION_LABELS)), INTERVENTION_LABELS, rotation=31, ha="right")
    ax.set_yticks(np.arange(len(TASK_ORDER)), [f"{d}–{h}" for d, h in TASK_ORDER])
    ax.tick_params(length=0, pad=1.3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("History intervention", loc="left", pad=17, fontweight="bold")
    ax.text(
        0.0,
        1.015,
        "Δ validation MSE (%) · frozen checkpoint · seed 2021",
        transform=ax.transAxes,
        fontsize=6.0,
        color="#555555",
        va="bottom",
    )


def add_panel_label(ax: plt.Axes, label: str, x: float) -> None:
    ax.text(x, 1.09, label, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="top", ha="left")


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

    primary_order = [(dataset, horizon) for dataset in ("ETTm1", "ETTm2")
                     for horizon in (96, 192, 336, 720)]
    transfer_order = [("ETTm1", 96), ("ETTm1", 720), ("ETTm2", 96), ("ETTm2", 720)]

    fig = plt.figure(figsize=(7.2, 4.55), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.48, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.105,
        right=0.985,
        bottom=0.115,
        top=0.895,
        wspace=0.34,
        hspace=0.72,
    )
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    draw_effect_rows(
        ax_a,
        primary,
        primary_order,
        title="Paired DHC gains",
        note="TimeRole vs RGSP-96 · test · 8 tasks × 3 seeds",
        xlim=(-1.0, 22.5),
        xticks=[0, 5, 10, 15, 20],
        xlabel="Gain over RGSP-96 (%)",
        show_legend=True,
    )
    draw_panel_b(ax_b, intervention_matrix)
    draw_effect_rows(
        ax_c,
        transfer,
        transfer_order,
        title="TimeXer transfer",
        note="TimeXer+DHC vs TimeXer-336 · 4 tasks × 3 seeds",
        xlim=(-2.5, 6.5),
        xticks=[-2, 0, 2, 4, 6],
        xlabel="Gain over TimeXer-336 (%)",
        show_legend=False,
    )

    add_panel_label(ax_a, "a", -0.19)
    add_panel_label(ax_b, "b", -0.16)
    add_panel_label(ax_c, "c", -0.16)

    svg_metadata = {
        "Title": "TimeRole evidence figure v2",
        "Description": "Paired DHC effects, distant-history interventions, and bounded backbone transfer",
        "Creator": "Python matplotlib",
    }
    pdf_metadata = {
        "Title": "TimeRole evidence figure v2",
        "Subject": "Paired DHC effects, distant-history interventions, and bounded backbone transfer",
        "Creator": "Python matplotlib",
    }
    fig.savefig(output_base.with_suffix(".svg"), metadata=svg_metadata)
    fig.savefig(output_base.with_suffix(".pdf"), metadata=pdf_metadata)
    fig.savefig(output_base.with_suffix(".png"), dpi=600, metadata=svg_metadata)
    fig.savefig(output_base.with_suffix(".tiff"), dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    write_source_data(primary, intervention, transfer, output_base)

    print(f"panel_a_rows={len(primary)}")
    print(f"panel_a_tasks={len(primary_order)}")
    print(f"panel_b_shape={intervention_matrix.shape[0]}x{intervention_matrix.shape[1]}")
    print(f"panel_c_rows={len(transfer)}")
    print(f"panel_c_wins_mse={(transfer['mse_improvement_pct'] > 0).sum()}/12")
    print(f"panel_c_wins_mae={(transfer['mae_improvement_pct'] > 0).sum()}/12")
    print(f"outputs={output_base}")


if __name__ == "__main__":
    main()
