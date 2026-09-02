"""Shared process and record helpers for TimeRole paper experiments."""

from __future__ import annotations

import json
import os
import re
import selectors
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PATTERN = re.compile(r"^VALIDATION_RESULT\s+(\{.*\})\s*$")
EVALUATION_PATTERN = re.compile(r"^EVALUATION_RESULT\s+(\{.*\})\s*$")


def replace(command: list[str], option: str, value: object) -> None:
    text = str(value)
    if option in command:
        command[command.index(option) + 1] = text
    else:
        command.extend((option, text))


def completed(path: Path, metric: str = "mse") -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "completed" and metric in payload


def atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def execute(
    command: list[str],
    log_path: Path,
    pattern: re.Pattern[str],
    gpu: int,
    timeout_seconds: int,
    cwd: Path = ROOT,
) -> tuple[int, dict[str, object] | None, float]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = None
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        deadline = started + timeout_seconds
        try:
            while True:
                for key, _ in selector.select(timeout=1.0):
                    line = key.fileobj.readline()
                    if line:
                        print(line, end="", flush=True)
                        handle.write(line)
                        handle.flush()
                        match = pattern.match(line.strip())
                        if match:
                            result = json.loads(match.group(1))
                if process.poll() is not None:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        handle.write(line)
                        match = pattern.match(line.strip())
                        if match:
                            result = json.loads(match.group(1))
                    return_code = int(process.returncode or 0)
                    break
                if time.monotonic() >= deadline:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    return_code = 124
                    break
        finally:
            selector.close()
            if process.poll() is None:
                process.terminate()
                process.wait()
    return return_code, result, time.monotonic() - started
