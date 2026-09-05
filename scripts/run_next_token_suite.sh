#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
STEPS=${STEPS:-4000}
WIDTH=${WIDTH:-128}
HEAD_SIZE=${HEAD_SIZE:-64}

if (( WIDTH % HEAD_SIZE != 0 )); then
    echo "WIDTH must be divisible by HEAD_SIZE" >&2
    exit 2
fi

heads=$((WIDTH / HEAD_SIZE))
configs=(adamw muon adamw_activation_control muon_activation_control)

for name in "${configs[@]}"; do
    run="results/next_token/${name}_width_${WIDTH}"
    mkdir -p "$run"
    "$PYTHON" -m experiments.next_token.train \
        --config "configs/next_token/$name.yaml" \
        --set "model.hidden_size=$WIDTH" \
        --set "model.num_attention_heads=$heads" \
        --set "training.steps=$STEPS" \
        --set "logging.tensorboard_dir=$run/tensorboard" \
        >"$run/train.log" 2>&1
done

touch "results/next_token/suite_width_${WIDTH}_complete"
