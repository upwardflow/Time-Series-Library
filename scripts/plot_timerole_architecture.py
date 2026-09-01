#!/usr/bin/env python3
"""Draw the concise TimeRole architecture with Python/matplotlib."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "timerole-matplotlib"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper/neurocomputing/figures/Fig1_TimeRole_Architecture"

INK = "#34383B"
MUTED = "#70767A"
BLUE = "#5F8FB6"
BLUE_LIGHT = "#DDEAF2"
ORANGE = "#D58A57"
ORANGE_LIGHT = "#F5E1D2"
GREEN = "#7C9A76"
GREEN_LIGHT = "#E2EBDD"
PURPLE = "#81729E"
PURPLE_LIGHT = "#E7E0EF"
CYAN_LIGHT = "#DDEEEF"
PANEL_BG = "#F8F9F9"


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    face: str,
    edge: str = INK,
    fontsize: float = 7.0,
    weight: str = "semibold",
    radius: float = 0.012,
    lw: float = 0.9,
    zorder: int = 3,
    rotation: float = 0,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize, color=INK,
            fontweight=weight, zorder=zorder + 1, linespacing=1.15,
            rotation=rotation, rotation_mode="anchor")
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    lw: float = 0.9,
    style: str = "-|>",
    dashed: bool = False,
    connectionstyle: str = "arc3,rad=0",
    zorder: int = 2,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=8,
        linewidth=lw,
        color=color,
        linestyle=(0, (3, 2)) if dashed else "solid",
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)


def wave(ax: plt.Axes, x0: float, x1: float, y: float, *, output: bool = False) -> None:
    x = np.linspace(x0, x1, 90)
    colors = (BLUE, ORANGE, GREEN) if not output else (ORANGE, GREEN, BLUE)
    for idx, color in enumerate(colors):
        yy = y + (idx - 1) * 0.018 + 0.009 * np.sin(
            2 * np.pi * (3.0 + 0.45 * idx) * (x - x0) / (x1 - x0) + idx * 0.75
        )
        ax.plot(x, yy, color=color, lw=1.0, clip_on=False, zorder=4)


def panel_frame(ax: plt.Axes, xy: tuple[float, float], width: float, height: float) -> None:
    ax.add_patch(FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.008,rounding_size=0.018",
        facecolor="white", edgecolor="#9CA2A6", linewidth=0.9,
        linestyle=(0, (4, 3)), zorder=0,
    ))


def orthogonal_arrow(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    *,
    color: str = INK,
    lw: float = 0.9,
    dashed: bool = False,
) -> None:
    """Draw a sharp-corner polyline whose final segment carries an arrowhead."""
    if len(points) < 2:
        raise ValueError("orthogonal_arrow requires at least two points")
    if len(points) > 2:
        ax.plot(
            [point[0] for point in points[:-1]],
            [point[1] for point in points[:-1]],
            color=color, lw=lw,
            ls=(0, (3, 2)) if dashed else "solid", zorder=2,
        )
    arrow(ax, points[-2], points[-1], color=color, lw=lw, dashed=dashed)


def edge_label(
    ax: plt.Axes,
    x: float,
    y: float,
    label: str,
    *,
    color: str = INK,
    ha: str = "center",
) -> None:
    """Place a compact variable label without obscuring its connector."""
    ax.text(
        x, y, label,
        ha=ha, va="center", fontsize=5.5, color=color, zorder=6,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.25, "alpha": 0.92},
    )


def draw_hero(ax: plt.Axes) -> None:
    ax.add_patch(FancyBboxPatch(
        (0.040, 0.535), 0.92, 0.405,
        boxstyle="round,pad=0.012,rounding_size=0.035",
        facecolor="#EEF4F6", edgecolor="#A7ADB0", linewidth=1.1,
        linestyle=(0, (4, 2)), zorder=0,
    ))
    ax.text(0.058, 0.900, "TimeRole", fontsize=12, fontweight="bold", color="#202426")

    wave(ax, 0.065, 0.105, 0.735)
    ax.text(0.085, 0.682, "Input", ha="center", fontsize=7.2, fontweight="bold")
    arrow(ax, (0.105, 0.735), (0.135, 0.735))

    box(ax, (0.135, 0.682), 0.095, 0.105, "Role split", face=PURPLE_LIGHT,
        edge=PURPLE, fontsize=7.3)

    box(ax, (0.270, 0.775), 0.10, 0.064, "Recent", face=BLUE_LIGHT, edge=BLUE)
    box(ax, (0.270, 0.625), 0.10, 0.064, "Distant", face=ORANGE_LIGHT, edge=ORANGE)

    arrow(ax, (0.230, 0.735), (0.270, 0.807))
    arrow(ax, (0.230, 0.735), (0.270, 0.657))

    box(ax, (0.415, 0.752), 0.14, 0.11, "RGSP\nBase forecast",
        face=BLUE_LIGHT, edge=BLUE, fontsize=7.4)
    box(ax, (0.415, 0.602), 0.14, 0.11, "DHC\nCorrection",
        face=ORANGE_LIGHT, edge=ORANGE, fontsize=7.4)
    arrow(ax, (0.370, 0.807), (0.415, 0.807), color=BLUE)
    arrow(ax, (0.370, 0.657), (0.415, 0.657), color=ORANGE)
    arrow(ax, (0.320, 0.775), (0.450, 0.712), color=BLUE, dashed=True)

    box(ax, (0.600, 0.775), 0.095, 0.064, r"Base $\widehat{\mathbf{Y}}^{b}$",
        face=CYAN_LIGHT, edge=BLUE, fontsize=7.0)
    box(ax, (0.600, 0.625), 0.095, 0.064, r"$\Delta\mathbf{Y}$",
        face=CYAN_LIGHT, edge=ORANGE, fontsize=7.0)
    arrow(ax, (0.555, 0.807), (0.600, 0.807), color=BLUE)
    arrow(ax, (0.555, 0.657), (0.600, 0.657), color=ORANGE)

    sum_center = (0.745, 0.735)
    ax.add_patch(Circle(sum_center, 0.024, facecolor="white", edgecolor=INK, lw=1.0, zorder=3))
    ax.text(*sum_center, "+", ha="center", va="center", fontsize=12, color=INK, zorder=4)
    arrow(ax, (0.695, 0.807), (0.727, 0.749), color=BLUE)
    arrow(ax, (0.695, 0.657), (0.727, 0.721), color=ORANGE)
    box(ax, (0.805, 0.680), 0.038, 0.11, "De-norm", face=PURPLE_LIGHT,
        edge=PURPLE, fontsize=6.8, rotation=270)
    arrow(ax, (0.769, 0.735), (0.805, 0.735))
    arrow(ax, (0.843, 0.735), (0.895, 0.735))
    wave(ax, 0.900, 0.940, 0.735, output=True)
    ax.text(0.920, 0.682, "Forecast", ha="center", fontsize=7.2, fontweight="bold")
    ax.text(0.055, 0.552, "a", fontsize=9.5, fontweight="bold")


def draw_rgsp(ax: plt.Axes) -> None:
    panel_frame(ax, (0.055, 0.055), 0.445, 0.405)
    ax.text(0.068, 0.432, "b", fontsize=9.5, fontweight="bold")
    ax.text(0.092, 0.432, "RGSP", fontsize=9.2, fontweight="bold")

    box(ax, (0.068, 0.215), 0.048, 0.070, r"$\mathbf{X}^{r}$",
        face=BLUE_LIGHT, edge=BLUE, fontsize=9.0)
    box(ax, (0.142, 0.185), 0.045, 0.130, "Decomp.",
        face=GREEN_LIGHT, edge=GREEN, fontsize=6.8, rotation=270)
    arrow(ax, (0.116, 0.250), (0.142, 0.250), color=BLUE)

    box(ax, (0.215, 0.275), 0.075, 0.080, "Dual-scale\npatches",
        face=BLUE_LIGHT, edge=BLUE, fontsize=7.3)
    box(ax, (0.315, 0.255), 0.043, 0.120, "Bi-Mamba",
        face=PURPLE_LIGHT, edge=PURPLE, fontsize=6.3, rotation=270)
    box(ax, (0.375, 0.250), 0.040, 0.130, "Var. graph",
        face=GREEN_LIGHT, edge=GREEN, fontsize=6.0, rotation=270)
    orthogonal_arrow(ax, [(0.187, 0.278), (0.202, 0.278), (0.202, 0.315), (0.215, 0.315)], color=BLUE)
    edge_label(ax, 0.198, 0.297, r"$\mathbf{S}^{r}$", color=BLUE, ha="right")
    arrow(ax, (0.290, 0.315), (0.315, 0.315), color=BLUE)
    edge_label(ax, 0.303, 0.330, r"$\mathbf{Z}^{(s,0)}$", color=BLUE)
    arrow(ax, (0.358, 0.315), (0.375, 0.315), color=PURPLE)
    edge_label(ax, 0.367, 0.330, r"$\mathbf{U}^{(s)}$", color=PURPLE)

    box(ax, (0.235, 0.075), 0.045, 0.130, "Trend proj.",
        face=ORANGE_LIGHT, edge=ORANGE, fontsize=6.1, rotation=270)
    orthogonal_arrow(ax, [(0.187, 0.222), (0.205, 0.222), (0.205, 0.140), (0.235, 0.140)], color=ORANGE)
    edge_label(ax, 0.199, 0.182, r"$\mathbf{T}^{r}$", color=ORANGE, ha="right")

    merge = (0.456, 0.315)
    ax.add_patch(Circle(merge, 0.016, facecolor="white", edgecolor=INK, lw=0.8, zorder=3))
    ax.text(*merge, "+", ha="center", va="center", fontsize=9, zorder=4)
    arrow(ax, (0.415, 0.315), (0.440, 0.315), color=GREEN)
    edge_label(ax, 0.428, 0.330, r"$\widehat{\mathbf{Y}}^{\mathrm{se}}$", color=GREEN)
    orthogonal_arrow(ax, [(0.280, 0.140), (0.430, 0.140), (0.430, 0.315), (0.440, 0.315)], color=ORANGE)
    edge_label(ax, 0.355, 0.155, r"$\widehat{\mathbf{Y}}^{\mathrm{tr}}$", color=ORANGE)
    arrow(ax, (0.472, 0.315), (0.483, 0.315), color=BLUE)
    ax.text(0.485, 0.315, r"$\widehat{\mathbf{Y}}^{b}$", ha="left", va="center", fontsize=8.0,
            color=BLUE, fontweight="bold")


def draw_dhc(ax: plt.Axes) -> None:
    panel_frame(ax, (0.520, 0.055), 0.425, 0.405)
    ax.text(0.533, 0.432, "c", fontsize=9.5, fontweight="bold")
    ax.text(0.557, 0.432, "DHC", fontsize=9.2, fontweight="bold")

    box(ax, (0.535, 0.285), 0.060, 0.070, r"$\mathbf{X}^{d}$",
        face=ORANGE_LIGHT, edge=ORANGE, fontsize=9.0)
    box(ax, (0.535, 0.145), 0.060, 0.070, r"$\mathbf{X}^{r}$",
        face=BLUE_LIGHT, edge=BLUE, fontsize=9.0)
    box(ax, (0.620, 0.275), 0.050, 0.100, "AvgPool",
        face=GREEN_LIGHT, edge=GREEN, fontsize=6.8, rotation=270)
    box(ax, (0.620, 0.105), 0.050, 0.130, "Recent state",
        face=GREEN_LIGHT, edge=GREEN, fontsize=6.3, rotation=270)
    arrow(ax, (0.595, 0.320), (0.620, 0.320), color=ORANGE)
    edge_label(ax, 0.608, 0.335, r"$\widetilde{\mathbf{X}}^{d}$", color=ORANGE)
    arrow(ax, (0.595, 0.180), (0.620, 0.180), color=BLUE)
    edge_label(ax, 0.608, 0.195, r"$\widetilde{\mathbf{X}}^{r}$", color=BLUE)

    box(ax, (0.725, 0.120), 0.055, 0.260, "Shared decoder",
        face=PURPLE_LIGHT, edge=PURPLE, fontsize=6.8, rotation=270)
    arrow(ax, (0.670, 0.320), (0.725, 0.320), color=ORANGE)
    edge_label(ax, 0.698, 0.335, r"$\mathbf{h}^{m}$", color=ORANGE)
    arrow(ax, (0.670, 0.180), (0.725, 0.180), color=BLUE)
    edge_label(ax, 0.698, 0.195, r"$\mathbf{h}^{r}$", color=BLUE)

    box(ax, (0.815, 0.212), 0.055, 0.076, "Diff.",
        face="white", edge=INK, fontsize=7.6)
    arrow(ax, (0.780, 0.250), (0.815, 0.250))
    edge_label(ax, 0.798, 0.266, r"$\mathbf{y}^{\pm m}$")
    box(ax, (0.892, 0.212), 0.045, 0.076, "Gate",
        face=CYAN_LIGHT, edge=GREEN, fontsize=7.2)
    arrow(ax, (0.870, 0.250), (0.892, 0.250))
    edge_label(ax, 0.881, 0.266, r"$\boldsymbol{\delta}$")
    ax.text(0.915, 0.172, r"$\Delta\mathbf{Y}$", ha="center", fontsize=9.0,
            color=ORANGE, fontweight="bold")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="output base path without extension")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    configure_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    draw_hero(ax)
    draw_rgsp(ax)
    draw_dhc(ax)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

    metadata = {
        "Title": "TimeRole architecture",
        "Description": "Role split, recent base predictor, and distant-history correction",
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
    print(f"outputs={output}")


if __name__ == "__main__":
    main()
