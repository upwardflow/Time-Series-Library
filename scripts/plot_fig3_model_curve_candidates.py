#!/usr/bin/env python3
"""Render one eight-model forecast grid for every dataset–horizon candidate."""

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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "paper/neurocomputing/figures/fig3_curve_candidates/source_npz"
DEFAULT_OUTPUT = ROOT / "paper/neurocomputing/figures/fig3_curve_candidates"
DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather")
HORIZONS = (96, 192, 336, 720)
MODELS = ("TimeRole", "S-Mamba", "iTransformer", "TimeMixer", "MSGNet", "PatchTST", "TimesNet", "DLinear")
SLUGS = {
    "TimeRole": "graphmambacmrhm", "S-Mamba": "smamba",
    "iTransformer": "itransformer", "TimeMixer": "timemixer",
    "MSGNet": "msgnet", "PatchTST": "patchtst",
    "TimesNet": "timesnet", "DLinear": "dlinear",
}
TRUTH_COLOR = "#245B8A"
PRED_COLOR = "#D17A45"


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
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def load_profiles(input_dir: Path, dataset: str, horizon: int) -> dict[str, dict[str, np.ndarray]]:
    profiles = {}
    for model in MODELS:
        path = input_dir / f"{dataset.lower()}_h{horizon}_{SLUGS[model]}.npz"
        with np.load(path, allow_pickle=False) as data:
            profiles[model] = {key: data[key].copy() for key in data.files}
    reference = profiles[MODELS[0]]["target"]
    reference_context = profiles[MODELS[0]]["context"]
    origin = int(profiles[MODELS[0]]["origin"])
    for model in MODELS[1:]:
        if int(profiles[model]["origin"]) != origin:
            raise ValueError(f"origin mismatch for {dataset} H={horizon}: {model}")
        if not np.allclose(reference, profiles[model]["target"], rtol=1e-6, atol=1e-5):
            raise ValueError(f"target mismatch for {dataset} H={horizon}: {model}")
        if not np.allclose(reference_context, profiles[model]["context"], rtol=1e-6, atol=1e-5):
            raise ValueError(f"context mismatch for {dataset} H={horizon}: {model}")
    return profiles


def draw_candidate(dataset: str, horizon: int, profiles: dict[str, dict[str, np.ndarray]]) -> plt.Figure:
    context = np.asarray(profiles[MODELS[0]]["context"], dtype=float)
    truth = np.asarray(profiles[MODELS[0]]["target"], dtype=float)
    input_length = len(context)
    full_truth = np.concatenate([context, truth])
    all_values = [full_truth] + [
        np.concatenate([context, np.asarray(profiles[m]["prediction"], dtype=float)])
        for m in MODELS
    ]
    combined = np.concatenate(all_values)
    span = float(combined.max() - combined.min())
    padding = max(0.05 * span, 1e-6)
    ylim = (float(combined.min() - padding), float(combined.max() + padding))
    x = np.arange(input_length + horizon)

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.7), sharex=True, sharey=True, facecolor="white")
    for panel, (ax, model) in enumerate(zip(axes.flat, MODELS)):
        prediction = np.asarray(profiles[model]["prediction"], dtype=float)
        full_prediction = np.concatenate([context, prediction])
        ax.plot(x, full_truth, color=TRUTH_COLOR, lw=1.0, label="Ground truth", zorder=3)
        ax.plot(x, full_prediction, color=PRED_COLOR, lw=0.9, label="Prediction", zorder=2)
        ax.axvline(input_length - 0.5, color="#85898D", lw=0.55, ls="--", zorder=1)
        ax.set_xlim(0, input_length + horizon - 1)
        ax.set_ylim(*ylim)
        ax.yaxis.grid(True, color="#E5E7E9", lw=0.4, zorder=-3)
        ax.tick_params(length=2.3, color="#777777")
        ax.set_title(f"({chr(97 + panel)}) {model}", pad=4)
        if panel // 4 == 1:
            ax.set_xlabel("Time step")
        if panel % 4 == 0:
            ax.set_ylabel("Standardized value")

    handles = [
        Line2D([0], [0], color=TRUTH_COLOR, lw=1.3, label="Ground truth"),
        Line2D([0], [0], color=PRED_COLOR, lw=1.2, label="Prediction"),
    ]
    fig.suptitle(
        f"{dataset} · input length L = {input_length} · forecast horizon H = {horizon}",
        fontsize=9, fontweight="bold", y=0.99,
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2,
               handlelength=2.4, columnspacing=1.4)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.12, top=0.86, wspace=0.18, hspace=0.32)
    return fig


def source_rows(dataset: str, horizon: int, profiles: dict[str, dict[str, np.ndarray]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    context = profiles[MODELS[0]]["context"]
    truth = np.concatenate([context, profiles[MODELS[0]]["target"]])
    input_length = len(context)
    origin = int(profiles[MODELS[0]]["origin"])
    for step, value in enumerate(truth):
        rows.append({"dataset": dataset, "horizon": horizon, "origin": origin,
                     "input_length": input_length, "step": step,
                     "phase": "input" if step < input_length else "forecast",
                     "series": "Ground truth", "value_standardized": float(value)})
    for model in MODELS:
        full_prediction = np.concatenate([context, profiles[model]["prediction"]])
        for step, value in enumerate(full_prediction):
            rows.append({"dataset": dataset, "horizon": horizon, "origin": origin,
                         "input_length": input_length, "step": step,
                         "phase": "input" if step < input_length else "forecast",
                         "series": model, "value_standardized": float(value)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="*", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--horizons", nargs="*", type=int, choices=HORIZONS, default=list(HORIZONS))
    args = parser.parse_args()
    input_dir = args.input if args.input.is_absolute() else ROOT / args.input
    output_dir = args.output if args.output.is_absolute() else ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    data_rows: list[dict[str, object]] = []
    book_path = output_dir / "Fig3_Model_Curve_Candidate_Book.pdf"
    with PdfPages(book_path, metadata={"Title": "Figure 3 model-curve candidates", "Creator": "Python matplotlib"}) as book:
        for dataset in args.datasets:
            for horizon in args.horizons:
                profiles = load_profiles(input_dir, dataset, horizon)
                fig = draw_candidate(dataset, horizon, profiles)
                base = output_dir / f"Fig3_Candidate_{dataset}_H{horizon}"
                fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
                fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
                fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
                book.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                data_rows.extend(source_rows(dataset, horizon, profiles))

    for dataset in args.datasets:
        available = [horizon for horizon in args.horizons
                     if (output_dir / f"Fig3_Candidate_{dataset}_H{horizon}.png").exists()]
        if not available:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(10.6, 5.8), facecolor="white")
        for ax in axes.flat:
            ax.set_axis_off()
        for ax, horizon in zip(axes.flat, available):
            image = plt.imread(output_dir / f"Fig3_Candidate_{dataset}_H{horizon}.png")
            ax.imshow(image)
            ax.set_title(f"H = {horizon}", fontsize=9, fontweight="bold", pad=2)
        fig.suptitle(f"{dataset} · Figure 3 candidate overview", fontsize=11,
                     fontweight="bold", y=0.995)
        fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.94,
                            wspace=0.015, hspace=0.08)
        overview = output_dir / f"Fig3_Candidate_Overview_{dataset}"
        fig.savefig(overview.with_suffix(".png"), dpi=220, bbox_inches="tight")
        fig.savefig(overview.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

    pd.DataFrame(data_rows).to_csv(output_dir / "candidate_source_data.csv", index=False)
    print(f"candidate_figures={len(args.datasets) * len(args.horizons)}")
    print(f"candidate_book={book_path}")


if __name__ == "__main__":
    main()
