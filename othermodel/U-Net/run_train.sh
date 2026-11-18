#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${DATASET_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/experiments/click_unet_em_domains}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WORKERS="${WORKERS:-4}"
LR="${LR:-0.001}"
CROP_SIZE="${CROP_SIZE:-512}"
BASE_CHANNELS="${BASE_CHANNELS:-32}"
NUM_POINTS="${NUM_POINTS:-15}"
CLICK_SIGMA="${CLICK_SIGMA:-10}"
DEVICE="${DEVICE:-cuda}"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash run_train.sh --dataset-root /path/to/Skeletal_muscle

Optional:
  --output-root PATH
  --epochs N
  --batch-size N
  --workers N
  --lr VALUE
  --crop-size N
  --base-channels N
  --num-points N
  --click-sigma VALUE
  --device cuda|cpu
  --domains D1 D2 ...
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --crop-size) CROP_SIZE="$2"; shift 2 ;;
    --base-channels) BASE_CHANNELS="$2"; shift 2 ;;
    --num-points) NUM_POINTS="$2"; shift 2 ;;
    --click-sigma) CLICK_SIGMA="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --domains)
      EXTRA_ARGS+=("--domains")
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        EXTRA_ARGS+=("$1")
        shift
      done
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 2 ;;
  esac
done

if [[ -z "$DATASET_ROOT" ]]; then
  echo "ERROR: --dataset-root is required."
  usage
  exit 2
fi

cd "$ROOT"
python train_click_unet_em_domains.py \
  --dataset-root "$DATASET_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --lr "$LR" \
  --crop-size "$CROP_SIZE" \
  --base-channels "$BASE_CHANNELS" \
  --num-points "$NUM_POINTS" \
  --click-sigma "$CLICK_SIGMA" \
  --device "$DEVICE" \
  "${EXTRA_ARGS[@]}"
