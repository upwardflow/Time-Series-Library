#!/usr/bin/env python3
"""Periodically refresh SCSD summaries until the tmux runner exits."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINALIZER = ROOT / "scripts" / "finalize_graphmamba_scsd.py"


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def refresh() -> int:
    return subprocess.run([sys.executable, str(FINALIZER)], cwd=ROOT).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    if args.pid < 1 or args.interval < 10:
        parser.error("pid must be positive and interval must be at least 10 seconds")
    while process_alive(args.pid):
        if refresh():
            return 1
        time.sleep(args.interval)
    return refresh()


if __name__ == "__main__":
    raise SystemExit(main())
