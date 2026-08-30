#!/usr/bin/env bash
set -euo pipefail
configs=(adamw muon muon_inverse_half soap_muon_shared joint_muon_p0 joint_muon_p05)
for name in "${configs[@]}"; do
    mkdir -p "results/$name"
    python3 experiments/train.py --config "configs/$name.yaml" \
        >"results/$name/train.log" 2>&1
done
touch results/suite_complete

