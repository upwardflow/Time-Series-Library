#!/usr/bin/env python3
"""Refresh the strict-audit report as the detached experiment queue advances."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "logs" / "timerole_strict_evidence" / "status.json"
FINALIZER = ROOT / "scripts" / "finalize_timerole_strict_evidence.py"


def main() -> int:
    last_signature = None
    while True:
        if STATUS.is_file():
            try:
                state = json.loads(STATUS.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                state = {}
            signature = (state.get("completed_new_jobs"), tuple(state.get("failed_jobs", [])))
            if signature != last_signature:
                subprocess.run([sys.executable, str(FINALIZER)], cwd=ROOT, check=False)
                last_signature = signature
            finished = (state.get("active_or_last_job") is None
                        and (state.get("completed_new_jobs", 0) + len(state.get("failed_jobs", [])) == 52))
            if finished:
                return 0
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
