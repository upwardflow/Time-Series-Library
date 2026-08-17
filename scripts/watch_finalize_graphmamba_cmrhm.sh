#!/usr/bin/env bash
set -u

cd /home/cwh/Time-Series-Library || exit 1
status_file="logs/graphmamba_cmrhm_six_dataset_final/status.json"
watch_log="logs/graphmamba_cmrhm_six_dataset_final/finalizer.log"

while true; do
    if .venv/bin/python -c 'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p.get("completed_total")==24 else 1)' "$status_file" 2>/dev/null; then
        .venv/bin/python -u scripts/finalize_graphmamba_cmrhm_comparison.py >> "$watch_log" 2>&1
        exit $?
    fi
    if ! tmux has-session -t graphmamba-cmrhm-sixds-final 2>/dev/null; then
        echo "Training tmux ended before 24/24; finalizer stopped." >> "$watch_log"
        exit 1
    fi
    sleep 30
done
