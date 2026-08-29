#!/usr/bin/env python3
"""Run gated TimeXer Recent96/DHC/Raw336 experiments on ETTh1/ETTh2.

All variants receive aligned 336-point samples. TimeXerRecent restricts the
native backbone to the final 96 points, TimeXerHistoryCorrection additionally
compresses the preceding 240 points, and native TimeXer directly processes all
336 points. Held-out testing remains blocked until every requested validation
checkpoint is complete.
"""

from __future__ import annotations

import run_timemixer_dhc_threeway as core


core.DEFAULT_OUTPUT = (
    core.ROOT / "logs" / "timexer_etth_dhc_threeway_multiseed"
)
core.ALL_DATASETS = ("ETTh1", "ETTh2")
core.ALL_HORIZONS = (96, 192, 336, 720)
core.VARIANTS = (
    ("TimeXerRecent", "recent96"),
    ("TimeXerHistoryCorrection", "dhc"),
    ("TimeXer", "raw336"),
)
core.PROTOCOL_NAME = "timexer_etth_recent96_dhc240_raw336_official_aligned_v1"


# Frozen from the repository's official TimeXer ETTh scripts. Values omitted
# there use run.py defaults and are made explicit here for provenance.
PRESETS = {
    ("ETTh1", 96): {
        "d_model": 256,
        "d_ff": 2048,
        "e_layers": 1,
        "batch_size": 4,
    },
    ("ETTh1", 192): {
        "d_model": 128,
        "d_ff": 2048,
        "e_layers": 2,
        "batch_size": 4,
    },
    ("ETTh1", 336): {
        "d_model": 512,
        "d_ff": 1024,
        "e_layers": 1,
        "batch_size": 16,
    },
    ("ETTh1", 720): {
        "d_model": 256,
        "d_ff": 1024,
        "e_layers": 1,
        "batch_size": 16,
    },
    ("ETTh2", 96): {
        "d_model": 256,
        "d_ff": 1024,
        "e_layers": 1,
        "batch_size": 16,
    },
    ("ETTh2", 192): {
        "d_model": 256,
        "d_ff": 1024,
        "e_layers": 1,
        "batch_size": 32,
    },
    ("ETTh2", 336): {
        "d_model": 512,
        "d_ff": 1024,
        "e_layers": 2,
        "batch_size": 4,
    },
    ("ETTh2", 720): {
        "d_model": 256,
        "d_ff": 1024,
        "e_layers": 2,
        "batch_size": 16,
    },
}


def parse_args():
    args = _base_parse_args()
    if "--epochs" not in core.sys.argv:
        args.epochs = 10
    if "--patience" not in core.sys.argv:
        args.patience = 3
    return args


def build_validation_command(
    args,
    dataset: str,
    horizon: int,
    seed: int,
    model: str,
    label: str,
) -> list[str]:
    preset = PRESETS[(dataset, horizon)]
    candidate = core.candidate_name(dataset, horizon, label, seed)
    return [
        core.sys.executable,
        "-u",
        str(core.RUN_PY),
        "--task_name",
        "long_term_forecast",
        "--is_training",
        "1",
        "--root_path",
        str(core.ROOT / "dataset" / "ETT-small"),
        "--data_path",
        f"{dataset}.csv",
        "--data",
        dataset,
        "--model_id",
        f"{dataset}_336_{horizon}_{candidate}",
        "--model",
        model,
        "--seed",
        str(seed),
        "--features",
        "M",
        "--seq_len",
        "336",
        "--label_len",
        "48",
        "--pred_len",
        str(horizon),
        "--patch_len",
        "16",
        "--timerole_recent_len",
        "96",
        "--timerole_memory_pool",
        "16",
        "--timerole_hidden_dim",
        "32",
        "--e_layers",
        str(preset["e_layers"]),
        "--factor",
        "3",
        "--enc_in",
        "7",
        "--dec_in",
        "7",
        "--c_out",
        "7",
        "--d_model",
        str(preset["d_model"]),
        "--d_ff",
        str(preset["d_ff"]),
        "--n_heads",
        "8",
        "--batch_size",
        str(preset["batch_size"]),
        "--learning_rate",
        "0.0001",
        "--train_epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--num_workers",
        str(args.num_workers),
        "--use_norm",
        "1",
        "--gpu",
        "0",
        "--des",
        candidate,
        "--itr",
        "1",
        "--test_after_train",
        "0",
        "--checkpoints",
        str(args.output_dir / "checkpoints"),
    ]


def setting_from_command(command: list[str]) -> str:
    option = core.option
    return (
        f"{option(command, '--task_name')}_{option(command, '--model_id')}_"
        f"{option(command, '--model')}_{option(command, '--data')}_"
        f"ft{option(command, '--features')}_sl{option(command, '--seq_len')}_"
        f"ll{option(command, '--label_len')}_pl{option(command, '--pred_len')}_"
        f"dm{option(command, '--d_model')}_nh{option(command, '--n_heads')}_"
        f"el{option(command, '--e_layers')}_dl1_df{option(command, '--d_ff')}_"
        f"expand2_dc4_fc{option(command, '--factor')}_ebtimeF_dtTrue_"
        f"{option(command, '--des')}_0"
    )


def write_protocol(args) -> None:
    payload = {
        "protocol": core.PROTOCOL_NAME,
        "created_before_test": True,
        "datasets": list(args.datasets),
        "horizons": list(args.horizons),
        "seeds": list(args.seeds),
        "variants": [label for _, label in core.VARIANTS],
        "aligned_seq_len": 336,
        "recent_len": 96,
        "old_len": 240,
        "memory_pool": 16,
        "hidden_dim": 32,
        "patch_len": 16,
        "presets": {
            f"{dataset}_p{horizon}": preset
            for (dataset, horizon), preset in PRESETS.items()
        },
        "learning_rate": 1e-4,
        "epochs": args.epochs,
        "patience": args.patience,
        "checkpoint_selection": "validation_best_mse",
        "metric": "element_weighted_mse_mae_v1",
        "mutability_envelope": [
            str(args.output_dir),
            "scripts/run_timexer_etth_dhc_threeway.py",
        ],
        "frozen_files": [
            "models/TimeXer.py",
            "models/TimeXerRecent.py",
            "models/TimeXerHistoryCorrection.py",
        ],
        "stop_conditions": [
            "non-finite loss or metric",
            "command/metadata mismatch within a paired cell",
            "missing or corrupt checkpoint",
            "GPU out-of-memory",
        ],
    }
    core.atomic_write_json(args.output_dir / "protocol.json", payload)


_base_parse_args = core.parse_args
core.parse_args = parse_args
core.build_validation_command = build_validation_command
core.setting_from_command = setting_from_command
core.write_protocol = write_protocol


if __name__ == "__main__":
    raise SystemExit(core.main())
