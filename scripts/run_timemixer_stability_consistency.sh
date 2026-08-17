#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$workspace_dir"

mkdir -p logs/q2_timemixer_stability/checkpoints logs/q2_timemixer_stability/logs

run_one() {
    local dataset="$1"
    local data_name="$2"
    local root_path="$3"
    local data_path="$4"
    local target="$5"
    local freq="$6"
    local channels="$7"
    local horizon="$8"
    local encoder_layers="$9"
    local epochs="${10}"
    local run_id="timemixer_stable_${dataset}_sl336_pl${horizon}_lr1e4_s2021"

    .venv/bin/python -u run.py \
        --task_name long_term_forecast --is_training 1 \
        --root_path "$root_path" --data_path "$data_path" \
        --model_id "$run_id" --model TimeMixer --seed 2021 \
        --data "$data_name" --features M --target "$target" --freq "$freq" \
        --seq_len 336 --label_len 0 --pred_len "$horizon" \
        --enc_in "$channels" --dec_in "$channels" --c_out "$channels" \
        --d_model 16 --d_ff 32 --n_heads 4 --e_layers "$encoder_layers" \
        --d_layers 1 --factor 3 --dropout 0.1 --batch_size 32 \
        --learning_rate 0.0001 --train_epochs "$epochs" --patience 3 \
        --lradj type1 --num_workers 0 --gpu 0 --des "$run_id" --itr 1 \
        --checkpoints logs/q2_timemixer_stability/checkpoints \
        --test_after_train 1 --down_sampling_layers 3 \
        --down_sampling_window 2 --down_sampling_method avg \
        2>&1 | tee "logs/q2_timemixer_stability/logs/${run_id}.log"
}

run_one ettm2 ETTm2 dataset/ETT-small ETTm2.csv OT t 7 720 2 10
run_one weather custom dataset/weather weather.csv "CO2 (ppm)" t 21 336 3 20
