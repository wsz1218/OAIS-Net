#!/usr/bin/env bash
set -euo pipefail

SAM2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-}"
CHECKPOINT="${CHECKPOINT:-$SAM2_ROOT/checkpoints/sam2.1_hiera_base_plus.pt}"
DATA_ROOT="${DATA_ROOT:-$SAM2_ROOT/em_data/oais_semantic_domains}"
EXP_ROOT="${EXP_ROOT:-$SAM2_ROOT/sam2_logs/em_bplus_semantic_domains}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RESOLUTION="${RESOLUTION:-1024}"
MULTIPLIER="${MULTIPLIER:-20}"
NUM_GPUS="${NUM_GPUS:-1}"
DOMAINS=()
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash em_tools/run_em_domain_finetune.sh --dataset-root /path/to/Skeletal_muscle

Optional arguments:
  --checkpoint PATH       Official SAM2.1 checkpoint, default: checkpoints/sam2.1_hiera_base_plus.pt
  --data-root PATH        Converted SAM2 data root, default: em_data/oais_semantic_domains
  --exp-root PATH         Per-domain log/checkpoint root, default: sam2_logs/em_bplus_semantic_domains
  --epochs N              Default: 40
  --batch-size N          Default: 1
  --num-workers N         Default: 4
  --resolution N          Default: 1024
  --multiplier N          Dataset repeat factor per epoch, default: 20
  --num-gpus N            Default: 1
  --domains D1 D2 ...     Default: the four paper domains
  --skip-data-prepare
  --skip-config
  --skip-train
  --dry-run

Expected checkpoints:
  <exp-root>/domain_2_cells/checkpoints/checkpoint.pt
  <exp-root>/domain_3_muscle/checkpoints/checkpoint.pt
  <exp-root>/domain_4_fuzzy_blobs/checkpoints/checkpoint.pt
  <exp-root>/domain_6_Lucchi/checkpoints/checkpoint.pt
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --exp-root) EXP_ROOT="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --resolution) RESOLUTION="$2"; shift 2 ;;
    --multiplier) MULTIPLIER="$2"; shift 2 ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --domains)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        DOMAINS+=("$1")
        shift
      done
      ;;
    --skip-data-prepare|--skip-config|--skip-train|--dry-run)
      EXTRA_ARGS+=("$1")
      shift
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

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: SAM2 checkpoint not found: $CHECKPOINT"
  exit 1
fi

cmd=(
  python "$SAM2_ROOT/em_tools/train_em_domains.py"
  --dataset-root "$DATASET_ROOT"
  --checkpoint "$CHECKPOINT"
  --data-root "$DATA_ROOT"
  --exp-root "$EXP_ROOT"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --resolution "$RESOLUTION"
  --multiplier "$MULTIPLIER"
  --num-gpus "$NUM_GPUS"
)

if [[ ${#DOMAINS[@]} -gt 0 ]]; then
  cmd+=(--domains "${DOMAINS[@]}")
fi

cmd+=("${EXTRA_ARGS[@]}")

cd "$SAM2_ROOT"
"${cmd[@]}"

echo
echo "SAM2 per-domain fine-tuning finished or training was skipped."
echo "Checkpoint root:"
echo "  $EXP_ROOT"
