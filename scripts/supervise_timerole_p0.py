#!/usr/bin/env python3
"""Gate and hand off the remaining TimeRole P0 stages through independent tmux sessions."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
BASE = ROOT / "logs" / "timerole_p0"
SUPERVISOR = BASE / "supervisor"
POLL_SECONDS = 60


def now() -> str:
    return datetime.now().astimezone().isoformat()


def write_state(status: str, stage: str, detail: str = "") -> None:
    SUPERVISOR.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "stage": stage, "detail": detail, "updated_at": now()}
    temporary = SUPERVISOR / "state.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(SUPERVISOR / "state.json")
    with (SUPERVISOR / "events.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def session_exists(name: str) -> bool:
    return subprocess.run(("tmux", "has-session", "-t", name), stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def wait_session(name: str, stage: str) -> None:
    while session_exists(name):
        write_state("waiting", stage, f"tmux session {name} is active")
        time.sleep(POLL_SECONDS)


def start_stage(name: str, command: list[str], stage: str) -> None:
    if session_exists(name):
        raise RuntimeError(f"refusing to reuse active tmux session: {name}")
    SUPERVISOR.mkdir(parents=True, exist_ok=True)
    exit_path = SUPERVISOR / f"{name}.exit"
    log_path = SUPERVISOR / f"{name}.log"
    if exit_path.exists():
        exit_path.unlink()
    shell = (
        f"cd {shlex.quote(str(ROOT))} && {shlex.join(command)} "
        f"> {shlex.quote(str(log_path))} 2>&1; "
        f"stage_code=$?; printf '%s\\n' \"$stage_code\" > {shlex.quote(str(exit_path))}; "
        "exit \"$stage_code\""
    )
    write_state("launching", stage, shlex.join(command))
    subprocess.run(("tmux", "new-session", "-d", "-s", name, "/bin/bash", "-lc", shell), check=True)
    wait_session(name, stage)
    if not exit_path.is_file():
        raise RuntimeError(f"{name} disappeared without an exit marker")
    code = int(exit_path.read_text(encoding="utf-8").strip())
    if code != 0:
        raise RuntimeError(f"{name} exited with code {code}; see {log_path}")


def require_json(path: Path, expected: dict[str, object], stage: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mismatches = [f"{key}={payload.get(key)!r}, expected {value!r}" for key, value in expected.items()
                  if payload.get(key) != value]
    if mismatches:
        raise RuntimeError(f"{stage} gate failed: {'; '.join(mismatches)}")
    write_state("gate_passed", stage, str(path))


def json_matches(path: Path, expected: dict[str, object]) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(payload.get(key) == value for key, value in expected.items())


def main() -> int:
    try:
        closest_expected = {"status": "completed", "expected": 24, "verified": 24, "problems": []}
        closest_audit = BASE / "closest" / "formal" / "final" / "audit.json"
        if json_matches(closest_audit, closest_expected):
            write_state("gate_passed", "closest_finalize", f"resume: {closest_audit}")
        else:
            wait_session("timerole_p0_closest_formal", "closest_formal")
            start_stage("timerole_p0_closest_finalize", [str(PYTHON), "-u", "scripts/finalize_timerole_closest.py"], "closest_finalize")
            require_json(closest_audit, closest_expected, "closest_finalize")

        sensitivity_expected = {"status": "completed", "expected": 120, "completed": 120, "failed": 0,
                                "split": "validation", "test_accessed": False}
        sensitivity_status = BASE / "sensitivity" / "status.json"
        if json_matches(sensitivity_status, sensitivity_expected):
            write_state("gate_passed", "sensitivity", f"resume: {sensitivity_status}")
        else:
            start_stage("timerole_p0_sensitivity", [str(PYTHON), "-u", "scripts/run_timerole_p0_sensitivity.py",
                        "--gpu", "0", "--epochs", "100", "--patience", "6", "--timeout-seconds", "7200"], "sensitivity")
            require_json(sensitivity_status, sensitivity_expected, "sensitivity")

        # A separately scheduled complete-grid test may share GPU 0.  Preserve
        # the preregistered single-job runtime protocol by waiting rather than
        # launching an ECL/Solar stage concurrently with that tmux session.
        wait_session("timerole_history_length_test", "external_history_length_test")

        for dataset, label in (("electricity", "ecl"), ("solar", "solar")):
            pilot_expected = {"status": "completed", "phase": "pilot", "dataset": dataset, "expected": 8,
                              "completed": 8, "failed": 0, "test_accessed": False}
            pilot_status = BASE / "ecl_solar" / "pilot" / "status.json"
            if json_matches(pilot_status, pilot_expected):
                write_state("gate_passed", f"{label}_pilot", f"resume: {pilot_status}")
            else:
                start_stage(f"timerole_p0_{label}_pilot", [str(PYTHON), "-u", "scripts/run_timerole_p0_ecl_solar.py",
                            "--phase", "pilot", "--datasets", dataset, "--gpu", "0", "--pilot-epochs", "2",
                            "--timeout-seconds", "21600"], f"{label}_pilot")
                require_json(pilot_status, pilot_expected, f"{label}_pilot")

            formal_expected = {"status": "completed", "phase": "formal", "dataset": dataset,
                               "expected": 96, "completed": 96, "failed": 0, "test_accessed": True}
            formal_status = BASE / "ecl_solar" / "formal" / "status.json"
            if json_matches(formal_status, formal_expected):
                write_state("gate_passed", f"{label}_formal", f"resume: {formal_status}")
            else:
                start_stage(f"timerole_p0_{label}_formal", [str(PYTHON), "-u", "scripts/run_timerole_p0_ecl_solar.py",
                            "--phase", "formal", "--datasets", dataset, "--gpu", "0",
                            "--timeout-seconds", "21600"], f"{label}_formal")
                require_json(formal_status, formal_expected, f"{label}_formal")

            start_stage(f"timerole_p0_{label}_finalize", [str(PYTHON), "-u", "scripts/finalize_timerole_p0_ecl_solar.py",
                        "--dataset", dataset], f"{label}_finalize")
            require_json(BASE / "ecl_solar" / "formal" / "final" / dataset / "audit.json",
                         {"status": "completed", "dataset": dataset, "expected": 96,
                          "verified": 96, "problems": [], "split": "test", "test_accessed": True}, f"{label}_finalize")

        write_state("completed", "solar_finalize", "all preregistered stages and audits passed")
        return 0
    except Exception as exc:
        write_state("blocked", "supervisor", str(exc))
        return 1
    except KeyboardInterrupt:
        write_state("interrupted", "supervisor", "external_keyboard_interrupt; no automatic retry")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
