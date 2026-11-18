#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${DATASET_ROOT:-}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/experiments/click_unet_em_domains}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/test_result/ClickUNet_EM}"
TARGET_CLICKS="${TARGET_CLICKS:-1,3,5,10,15}"
TARGET_IOUS="${TARGET_IOUS:-0.85,0.88,0.90,0.93}"
MAX_CLICKS="${MAX_CLICKS:-15}"
VISUALIZE_MAX_IMAGES="${VISUALIZE_MAX_IMAGES:-3}"
DEVICE="${DEVICE:-cuda}"
SAVE_PATENT="${SAVE_PATENT:-0}"
PATENT_STEP="${PATENT_STEP:-10}"
PATENT_TITLE="${PATENT_TITLE:-TEM interactive segmentation result}"
PATENT_REFERENCE_ROOT="${PATENT_REFERENCE_ROOT:-}"
PATENT_REFERENCE_LABEL="${PATENT_REFERENCE_LABEL:-Base model}"
PATENT_METHOD_LABEL="${PATENT_METHOD_LABEL:-Interactive U-Net}"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash run_eval.sh --dataset-root /path/to/Skeletal_muscle

Optional:
  --checkpoint-root PATH
  --output-root PATH
  --target-clicks CSV
  --target-ious CSV
  --max-clicks N
  --visualize-max-images N
  --device cuda|cpu
  --save-patent
  --patent-step N
  --patent-title TEXT
  --patent-reference-root PATH
  --domains D1 D2 ...
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --checkpoint-root) CHECKPOINT_ROOT="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --target-clicks) TARGET_CLICKS="$2"; shift 2 ;;
    --target-ious) TARGET_IOUS="$2"; shift 2 ;;
    --max-clicks) MAX_CLICKS="$2"; shift 2 ;;
    --visualize-max-images) VISUALIZE_MAX_IMAGES="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --save-patent) SAVE_PATENT=1; shift ;;
    --patent-step) PATENT_STEP="$2"; shift 2 ;;
    --patent-title) PATENT_TITLE="$2"; shift 2 ;;
    --patent-reference-root) PATENT_REFERENCE_ROOT="$2"; shift 2 ;;
    --patent-reference-label) PATENT_REFERENCE_LABEL="$2"; shift 2 ;;
    --patent-method-label) PATENT_METHOD_LABEL="$2"; shift 2 ;;
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
cmd=(
  python evaluate_click_unet_em_domains.py
  --dataset-root "$DATASET_ROOT"
  --checkpoint-root "$CHECKPOINT_ROOT"
  --output-root "$OUTPUT_ROOT"
  --target-clicks "$TARGET_CLICKS"
  --target-ious "$TARGET_IOUS"
  --max-clicks "$MAX_CLICKS"
  --visualize-max-images "$VISUALIZE_MAX_IMAGES"
  --device "$DEVICE"
  --patent-step "$PATENT_STEP"
  --patent-title "$PATENT_TITLE"
  --patent-reference-label "$PATENT_REFERENCE_LABEL"
  --patent-method-label "$PATENT_METHOD_LABEL"
)

if [[ "$SAVE_PATENT" == "1" ]]; then
  cmd+=(--save-patent)
fi
if [[ -n "$PATENT_REFERENCE_ROOT" ]]; then
  cmd+=(--patent-reference-root "$PATENT_REFERENCE_ROOT")
fi

cmd+=("${EXTRA_ARGS[@]}")
"${cmd[@]}"
