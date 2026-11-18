#!/usr/bin/env bash
set -euo pipefail

SAM2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-}"
CHECKPOINT="${CHECKPOINT:-$SAM2_ROOT/sam2_logs/em_bplus_semantic_finetune/checkpoints/checkpoint.pt}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"
CHECKPOINT_PATTERN="${CHECKPOINT_PATTERN:-__DOMAIN__/checkpoints/checkpoint.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SAM2_ROOT/test_result/SAM2_EM}"
MERGE_STRATEGY="${MERGE_STRATEGY:-prompt_set}"
MODEL_CFG="${MODEL_CFG:-configs/sam2.1/sam2.1_hiera_b+.yaml}"
MAX_CLICKS="${MAX_CLICKS:-15}"
TARGET_CLICKS="${TARGET_CLICKS:-1,3,5,10,15}"
TARGET_IOUS="${TARGET_IOUS:-0.85,0.88,0.90,0.93}"
VISUALIZE_MAX_IMAGES="${VISUALIZE_MAX_IMAGES:-3}"
DEVICE="${DEVICE:-cuda}"
SAVE_PATENT="${SAVE_PATENT:-0}"
PATENT_STEP="${PATENT_STEP:-10}"
PATENT_TITLE="${PATENT_TITLE:-TEM interactive segmentation result}"
PATENT_REFERENCE_ROOT="${PATENT_REFERENCE_ROOT:-}"
PATENT_REFERENCE_LABEL="${PATENT_REFERENCE_LABEL:-Base model}"
PATENT_METHOD_LABEL="${PATENT_METHOD_LABEL:-}"

usage() {
  cat <<'EOF'
Usage:
  bash em_tools/run_em_eval.sh --dataset-root /path/to/Skeletal_muscle --checkpoint /path/to/checkpoint.pt
  bash em_tools/run_em_eval.sh --dataset-root /path/to/Skeletal_muscle --checkpoint-root /path/to/em_bplus_semantic_domains

Optional arguments:
  --checkpoint-root PATH
  --checkpoint-pattern PATTERN
  --output-root PATH
  --merge-strategy prompt_set|positive_union
  --model-cfg PATH_RELATIVE_TO_SAM2_ROOT
  --device cuda|cpu
  --max-clicks N
  --target-clicks CSV
  --target-ious CSV
  --save-patent
  --patent-step N
  --patent-title TEXT
  --patent-reference-root PATH
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --checkpoint-root) CHECKPOINT_ROOT="$2"; shift 2 ;;
    --checkpoint-pattern) CHECKPOINT_PATTERN="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --merge-strategy) MERGE_STRATEGY="$2"; shift 2 ;;
    --model-cfg) MODEL_CFG="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --max-clicks) MAX_CLICKS="$2"; shift 2 ;;
    --target-clicks) TARGET_CLICKS="$2"; shift 2 ;;
    --target-ious) TARGET_IOUS="$2"; shift 2 ;;
    --save-patent) SAVE_PATENT=1; shift ;;
    --patent-step) PATENT_STEP="$2"; shift 2 ;;
    --patent-title) PATENT_TITLE="$2"; shift 2 ;;
    --patent-reference-root) PATENT_REFERENCE_ROOT="$2"; shift 2 ;;
    --patent-reference-label) PATENT_REFERENCE_LABEL="$2"; shift 2 ;;
    --patent-method-label) PATENT_METHOD_LABEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 2 ;;
  esac
done

if [[ -z "$DATASET_ROOT" ]]; then
  echo "ERROR: --dataset-root is required."
  usage
  exit 2
fi

if [[ -z "$CHECKPOINT_ROOT" && ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT"
  exit 1
fi

cd "$SAM2_ROOT"
cmd=(
  python em_tools/evaluate_em_domains.py
  --dataset-root "$DATASET_ROOT" \
  --model-cfg "$MODEL_CFG" \
  --output-root "$OUTPUT_ROOT" \
  --device "$DEVICE" \
  --merge-strategy "$MERGE_STRATEGY" \
  --max-clicks "$MAX_CLICKS" \
  --target-clicks "$TARGET_CLICKS" \
  --target-ious "$TARGET_IOUS" \
  --visualize-max-images "$VISUALIZE_MAX_IMAGES" \
  --patent-step "$PATENT_STEP" \
  --patent-title "$PATENT_TITLE" \
  --patent-reference-label "$PATENT_REFERENCE_LABEL"
)

if [[ -n "$CHECKPOINT_ROOT" ]]; then
  cmd+=(--checkpoint-root "$CHECKPOINT_ROOT" --checkpoint-pattern "$CHECKPOINT_PATTERN")
else
  cmd+=(--checkpoint "$CHECKPOINT")
fi

if [[ "$SAVE_PATENT" == "1" ]]; then
  cmd+=(--save-patent)
fi
if [[ -n "$PATENT_REFERENCE_ROOT" ]]; then
  cmd+=(--patent-reference-root "$PATENT_REFERENCE_ROOT")
fi
if [[ -n "$PATENT_METHOD_LABEL" ]]; then
  cmd+=(--patent-method-label "$PATENT_METHOD_LABEL")
fi

"${cmd[@]}"
