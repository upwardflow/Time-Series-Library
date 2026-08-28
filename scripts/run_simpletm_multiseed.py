#!/usr/bin/env python3
"""Complete SimpleTM seeds 2022/2023 and aggregate all three seeds."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"


def run(command: list[str]) -> None:
    print("STAGE", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> int:
    for seed in (2022, 2023):
        run([
            str(PYTHON), "-u", str(ROOT / "scripts/run_simpletm_etth1.py"),
            "--horizons", "96", "192", "336", "720",
            "--seed", str(seed), "--gpu", "0",
        ])
        run([
            str(PYTHON), "-u", str(ROOT / "scripts/run_simpletm_remaining.py"),
            "--datasets", "ETTh2", "ETTm1", "ETTm2", "weather",
            "--horizons", "96", "192", "336", "720",
            "--seed", str(seed), "--gpu", "0",
        ])
    run([str(PYTHON), "-u", str(ROOT / "scripts/finalize_simpletm_multiseed.py")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
