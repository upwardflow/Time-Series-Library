# GraphMamba SCSD validation operations

This directory contains the validation-only SCSD Phase-0 and Phase-1 runs.
No command in the active queue enables test evaluation.

## Frozen queue

- Phase 0: ETTh1/ETTh2, horizon 192, seed 2021, J and IS (4 runs).
- Phase 1: ETTm1/ETTh2/weather × horizons 96/720 × seeds 2021/2022/2023 × J/IS/IU/C/F (90 runs).
- D is represented by IS because both mean the same dual-scale, independent-shared configuration.
- IU parameter-matched controls are scheduled only after the raw IU/IS/SCSD parameter counts are known.
- SA, DC, full SCSD, TimeRole integration, and every test-split run remain gated.

## tmux

Session: `scsd_validation`

- Window 0: serial, resumable experiment queue.
- Window `aggregate`: refreshes `summary.csv` and `summary.json` every five minutes and once after the queue exits.

Useful commands:

```bash
tmux attach -t scsd_validation
tmux capture-pane -pt scsd_validation:0 -S -120
tmux capture-pane -pt scsd_validation:aggregate -S -80
```

To stop safely, send Ctrl-C to window 0. The current failed/interrupted run will
be recorded and the queue will stop without a silent retry:

```bash
tmux send-keys -t scsd_validation:0 C-c
```

To resume completed work after reviewing any incident, run the scheduler again;
completed JSON records are skipped. A failed record requires explicit
`--retry-failed`, which appends retry provenance to `incidents.jsonl`.

## Artifacts

- `manifest.json`: frozen scheduled tasks and source state.
- `records/*.json`: atomic per-run structured records.
- `raw_logs/*.log`: complete stdout/stderr for each run.
- `checkpoints/`: best-validation checkpoints.
- `audit/structure_audit.json`: Phase-0 object/shape/parameter audit.
- `incidents.jsonl`: append-only failures and explicit retries (created on first incident).
- `summary.csv`, `summary.json`: read-only aggregate status.

Each record stores the Git commit, dirty-state and source-file hashes, full
command and resolved configuration, validation metrics, best epoch, parameter
count, training duration, training/inference memory, inference latency, and
`test_accessed=false`.
