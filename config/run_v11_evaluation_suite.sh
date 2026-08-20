#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/kang/miniforge3/envs/env_isaaclab/bin/python"
CHECKPOINT="${1:?usage: run_v11_evaluation_suite.sh CHECKPOINT [OUTPUT_DIR] [v11|v10_4]}"
OUTPUT_DIR="${2:-$ROOT/results/v11_multiseed}"
EVAL_ENV="${3:-v11}"
SEEDS=(42 43 44 45 46)

case "$EVAL_ENV" in
    v11) ROUGH_ENV="v11_terrain" ;;
    v10_4) ROUGH_ENV="v10_4_terrain" ;;
    *) echo "unsupported evaluation environment: $EVAL_ENV" >&2; exit 2 ;;
esac

CHECKPOINT="$(realpath "$CHECKPOINT")"
mkdir -p "$OUTPUT_DIR"
cd "$ROOT"

rough_json=()
stairs_json=()
for seed in "${SEEDS[@]}"; do
    rough="$OUTPUT_DIR/rough_seed_${seed}.json"
    stairs="$OUTPUT_DIR/stairs_seed_${seed}.json"
    "$PYTHON" evaluate_v8.py \
        --headless \
        --env "$ROUGH_ENV" \
        --forward-axis +y \
        --terrain-level 4 \
        --checkpoint "$CHECKPOINT" \
        --output "$rough" \
        --seed "$seed" \
        --num_envs 64 \
        --seconds 8 \
        --warmup-seconds 1 \
        --speed 0.5 \
        >"$OUTPUT_DIR/rough_seed_${seed}.log" 2>&1 || true
    if [[ ! -f "$rough" ]]; then
        echo "missing rough result for seed $seed" >&2
        exit 1
    fi
    rough_json+=("$rough")

    "$PYTHON" evaluate_v11_stairs.py \
        --headless \
        --env "$EVAL_ENV" \
        --checkpoint "$CHECKPOINT" \
        --output "$stairs" \
        --seed "$seed" \
        --num_envs 64 \
        --seconds 12 \
        --speed 0.5 \
        >"$OUTPUT_DIR/stairs_seed_${seed}.log" 2>&1 || true
    if [[ ! -f "$stairs" ]]; then
        echo "missing stairs result for seed $seed" >&2
        exit 1
    fi
    stairs_json+=("$stairs")
done

"$PYTHON" aggregate_v11_multiseed.py \
    --scenario rough \
    --output "$OUTPUT_DIR/rough_aggregate.json" \
    "${rough_json[@]}"
"$PYTHON" aggregate_v11_multiseed.py \
    --scenario stairs \
    --output "$OUTPUT_DIR/stairs_aggregate.json" \
    "${stairs_json[@]}"

echo "evaluation_dir=$OUTPUT_DIR"
