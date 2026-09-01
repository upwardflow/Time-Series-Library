#!/usr/bin/env python3
"""Render the formal eight-model Figure 3 forecast case in Python."""

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
DEFAULT_INPUT = ROOT / "paper/neurocomputing/figures/fig3_curve_candidates/source_npz"
DEFAULT_OUTPUT = ROOT / "paper/neurocomputing/figures/Fig3_Model_Comparison_v2"
MODELS = ("TimeRole", "S-Mamba", "iTransformer", "TimeMixer", "MSGNet", "PatchTST", "TimesNet", "DLinear")
SLUGS = {
    "TimeRole": "graphmambacmrhm", "S-Mamba": "smamba",
    "iTransformer": "itransformer", "TimeMixer": "timemixer",
    "MSGNet": "msgnet", "PatchTST": "patchtst",
    "TimesNet": "timesnet", "DLinear": "dlinear",
}
TRUTH = "#245B8A"
PRED = "#D17A45"


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
        "xtick.labelsize": 6.1,
        "ytick.labelsize": 6.1,
        "legend.fontsize": 6.3,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def load_profiles(input_dir: Path, dataset: str, horizon: int) -> dict[str, dict[str, np.ndarray]]:
    profiles: dict[str, dict[str, np.ndarray]] = {}
    for model in MODELS:
        path = input_dir / f"{dataset.lower()}_h{horizon}_{SLUGS[model]}.npz"
        with np.load(path, allow_pickle=False) as data:
            profiles[model] = {key: data[key].copy() for key in data.files}
    reference = profiles[MODELS[0]]
    for model in MODELS:
        profile = profiles[model]
        if bool(profile["inverse_transformed"]):
            raise ValueError(f"{model} is not on the standardized scale")
        if int(profile["origin"]) != int(reference["origin"]):
            raise ValueError(f"origin mismatch: {model}")
        for key in ("context", "target"):
            if not np.allclose(profile[key], reference[key], rtol=1e-6, atol=1e-5):
                raise ValueError(f"{key} mismatch: {model}")
    if len(reference["context"]) != 96 or len(reference["target"]) != horizon:
        raise ValueError("unexpected input/forecast length")
    return profiles


def source_data(dataset: str, horizon: int, profiles: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    context = np.asarray(profiles[MODELS[0]]["context"], dtype=float)
    target = np.asarray(profiles[MODELS[0]]["target"], dtype=float)
    truth = np.concatenate([context, target])
    origin = int(profiles[MODELS[0]]["origin"])
    rows: list[dict[str, object]] = []
    for step, value in enumerate(truth):
        rows.append({
            "dataset": dataset, "horizon": horizon, "origin": origin,
            "input_length": len(context), "step": step,
            "phase": "input" if step < len(context) else "forecast",
            "series": "Ground truth", "value_standardized": float(value),
        })
    for model in MODELS:
        curve = np.concatenate([context, np.asarray(profiles[model]["prediction"], dtype=float)])
        for step, value in enumerate(curve):
            rows.append({
                "dataset": dataset, "horizon": horizon, "origin": origin,
                "input_length": len(context), "step": step,
                "phase": "input" if step < len(context) else "forecast",
                "series": model, "value_standardized": float(value),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", default="ETTh2", choices=("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"))
    parser.add_argument("--horizon", type=int, default=96, choices=(96, 192, 336, 720))
    args = parser.parse_args()
    input_dir = args.input if args.input.is_absolute() else ROOT / args.input
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    configure_style()
    profiles = load_profiles(input_dir, args.dataset, args.horizon)
    context = np.asarray(profiles[MODELS[0]]["context"], dtype=float)
    target = np.asarray(profiles[MODELS[0]]["target"], dtype=float)
    full_truth = np.concatenate([context, target])
    input_length = len(context)
    x = np.arange(input_length + args.horizon)
    curves = [full_truth] + [
        np.concatenate([context, np.asarray(profiles[m]["prediction"], dtype=float)])
        for m in MODELS
    ]
    values = np.concatenate(curves)
    pad = max(0.05 * float(np.ptp(values)), 1e-5)
    ylim = float(values.min() - pad), float(values.max() + pad)

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.72), sharex=True, sharey=True, facecolor="white")
    for idx, (ax, model) in enumerate(zip(axes.flat, MODELS)):
        prediction = np.concatenate([context, np.asarray(profiles[model]["prediction"], dtype=float)])
        ax.axvspan(0, input_length - 0.5, color="#F4F6F7", zorder=-4)
        ax.axvline(input_length - 0.5, color="#858A8E", lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.plot(x, full_truth, color=TRUTH, lw=1.05, zorder=4)
        ax.plot(x, prediction, color=PRED, lw=0.9, zorder=3)
        ax.set_xlim(0, input_length + args.horizon - 1)
        ax.set_ylim(*ylim)
        ax.set_xticks([0, 48, 96, 144, input_length + args.horizon - 1])
        ax.yaxis.grid(True, color="#E3E6E8", lw=0.42, zorder=-3)
        ax.tick_params(length=2.3, color="#777777")
        ax.spines["left"].set_color("#666666")
        ax.spines["bottom"].set_color("#666666")
        ax.set_title(f"({chr(97 + idx)}) {model}", pad=4)
        if idx // 4 == 1:
            ax.set_xlabel("Time step")
        if idx % 4 == 0:
            ax.set_ylabel("Standardized value")

    handles = [
        Line2D([0], [0], color=TRUTH, lw=1.35, label="Ground truth"),
        Line2D([0], [0], color=PRED, lw=1.2, label="Prediction"),
    ]
    fig.suptitle(
        f"{args.dataset} · input {input_length} · forecast {args.horizon}",
        fontsize=9, fontweight="bold", y=0.992,
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.947),
               ncol=2, handlelength=2.4, columnspacing=1.5)
    fig.text(0.195, 0.914, "Input", ha="center", fontsize=5.7, color="#707579")
    fig.text(0.285, 0.914, "Forecast", ha="center", fontsize=5.7, color="#707579")
    fig.subplots_adjust(left=0.077, right=0.993, bottom=0.12, top=0.855,
                        wspace=0.18, hspace=0.32)

    metadata = {
        "Title": "Eight-model forecast comparison",
        "Description": f"{args.dataset} standardized input-{input_length} forecast-{args.horizon} case",
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

    source = source_data(args.dataset, args.horizon, profiles)
    source.to_csv(output.with_name(output.name + "_source_data.csv"), index=False)
    print(f"source_rows={len(source)}")
    print(f"outputs={output}")


if __name__ == "__main__":
    main()
