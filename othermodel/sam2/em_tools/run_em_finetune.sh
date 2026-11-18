#!/usr/bin/env bash
set -euo pipefail

SAM2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-}"
DATA_DIR="${DATA_DIR:-$SAM2_ROOT/em_data/oais_semantic_train}"
CHECKPOINT="${CHECKPOINT:-$SAM2_ROOT/checkpoints/sam2.1_hiera_base_plus.pt}"
CONFIG_NAME="${CONFIG_NAME:-sam2.1_hiera_b+_EM_semantic_finetune.yaml}"
EXP_DIR="${EXP_DIR:-$SAM2_ROOT/sam2_logs/em_bplus_semantic_finetune}"
EPOCHS="${EPOCHS:-40}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_FRAMES="${NUM_FRAMES:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RESOLUTION="${RESOLUTION:-1024}"
NUM_GPUS="${NUM_GPUS:-1}"
SKIP_DATA_PREPARE="${SKIP_DATA_PREPARE:-0}"
SKIP_CONFIG="${SKIP_CONFIG:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash em_tools/run_em_finetune.sh --dataset-root /path/to/pi_seg/data/datasets

Optional arguments:
  --checkpoint PATH       SAM2.1 checkpoint, default: checkpoints/sam2.1_hiera_base_plus.pt
  --data-dir PATH         Converted SAM2 EM data output dir
  --exp-dir PATH          Training log/checkpoint output dir
  --epochs N              Default: 40
  --batch-size N          Default: 1
  --num-workers N         Default: 4
  --resolution N          Default: 1024
  --num-gpus N            Default: 1
  --skip-data-prepare
  --skip-config
  --skip-train

Equivalent environment variables are also supported:
  DATASET_ROOT, CHECKPOINT, DATA_DIR, EXP_DIR, EPOCHS, BATCH_SIZE,
  NUM_WORKERS, RESOLUTION, NUM_GPUS.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --exp-dir) EXP_DIR="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --resolution) RESOLUTION="$2"; shift 2 ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --skip-data-prepare) SKIP_DATA_PREPARE=1; shift ;;
    --skip-config) SKIP_CONFIG=1; shift ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 2 ;;
  esac
done

if [[ -z "$DATASET_ROOT" ]]; then
  echo "ERROR: --dataset-root is required on the server."
  usage
  exit 2
fi

if [[ "$SKIP_CONFIG" != "1" && ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: SAM2 checkpoint not found: $CHECKPOINT"
  exit 1
fi

cd "$SAM2_ROOT"

if [[ "$SKIP_DATA_PREPARE" != "1" ]]; then
  python em_tools/prepare_em_data.py \
    --scan-root "$DATASET_ROOT" \
    --output-dir "$DATA_DIR"
fi

if [[ "$SKIP_CONFIG" != "1" ]]; then
  python em_tools/make_em_config.py \
    --data-dir "$DATA_DIR" \
    --checkpoint "$CHECKPOINT" \
    --config-name "$CONFIG_NAME" \
    --experiment-log-dir "$EXP_DIR" \
    --epochs "$EPOCHS" \
    --num-workers "$NUM_WORKERS" \
    --num-frames "$NUM_FRAMES" \
    --batch-size "$BATCH_SIZE" \
    --resolution "$RESOLUTION" \
    --gpus-per-node "$NUM_GPUS"
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  python training/train.py \
    -c "configs/sam2.1_training/$CONFIG_NAME" \
    --use-cluster 0 \
    --num-gpus "$NUM_GPUS"
fi

echo
echo "SAM2-EM semantic fine-tuning finished or training was skipped."
echo "Expected checkpoint:"
echo "  $EXP_DIR/checkpoints/checkpoint.pt"
