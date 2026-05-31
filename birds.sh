#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
MODE="${MODE:-full}"
OUTPUT_DIR="${OUTPUT_DIR:-results/camera_ready_repro}"
TARGETS="${TARGETS:-all}"
SEED_START="${SEED_START:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
QUALITY_BATCH_SIZE="${QUALITY_BATCH_SIZE:-32}"
NUM_REF="${NUM_REF:--1}"

case "${MODE}" in
  smoke)
    NUM_SAMPLES="${NUM_SAMPLES:-4}"
    RUN_QUALITY="${RUN_QUALITY:-0}"
    ;;
  full)
    NUM_SAMPLES="${NUM_SAMPLES:-256}"
    RUN_QUALITY="${RUN_QUALITY:-1}"
    ;;
  *)
    echo "MODE must be smoke or full, got: ${MODE}" >&2
    exit 1
    ;;
esac

bird_target() {
  case "$1" in
    111|brown_headed_cowbird) echo "111" ;;
    222|fairy_tern) echo "222" ;;
    333|lucifer_hummingbird) echo "333" ;;
    444|scarlet_macaw) echo "444" ;;
    *) echo "Unknown bird target: $1" >&2; exit 1 ;;
  esac
}

require_file() {
  if [[ ! -e "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

join_csv() {
  local out=""
  for item in "$@"; do
    if [[ -n "${out}" ]]; then
      out="${out},${item}"
    else
      out="${item}"
    fi
  done
  echo "${out}"
}

if [[ "${TARGETS}" == "all" ]]; then
  target_list=(111 222 333 444)
  targets_arg="all"
  display_targets="111 222 333 444"
else
  normalized_targets="${TARGETS//,/ }"
  target_list=()
  for target in ${normalized_targets}; do
    target_list+=("$(bird_target "${target}")")
  done
  targets_arg="$(join_csv "${target_list[@]}")"
  display_targets="${target_list[*]}"
fi

launcher_dir="${OUTPUT_DIR}/_launcher/birds"
mkdir -p "${launcher_dir}"
exec > >(tee -a "${launcher_dir}/launcher.log") 2>&1

echo "birds_camera_ready_start=$(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "mode=${MODE} output_dir=${OUTPUT_DIR} targets=${display_targets}"
echo "num_samples=${NUM_SAMPLES} seed_start=${SEED_START} batch_size=${BATCH_SIZE}"

require_file "checkpoints/openai_imagenet.pt"
require_file "repro/assets/birds_filters/full_dataset.pt"
for target in "${target_list[@]}"; do
  require_file "repro/assets/birds_filters/class_${target}.pt"
  if [[ ! -e "repro/assets/birds_concepts/label_${target}.pt" &&
        ! -e "repro/assets/birds_concepts/rfm_label_${target}.pt" &&
        ! -e "repro/assets/birds_concepts/class_${target}.pt" ]]; then
    echo "Missing bird RFM concept for target ${target} under repro/assets/birds_concepts" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" repro/run_camera_ready.py \
  --task birds \
  --method na_rfm \
  --mode "${MODE}" \
  --targets "${targets_arg}" \
  --num-samples "${NUM_SAMPLES}" \
  --seed-start "${SEED_START}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --output-root "${OUTPUT_DIR}" \
  --python "${PYTHON_BIN}"

if [[ "${RUN_QUALITY}" == "1" ]]; then
  quality_summary="${OUTPUT_DIR}/birds/na_rfm/quality_summary.json"
  "${PYTHON_BIN}" repro/evaluate_saved_samples.py \
    --dataset=bird-species \
    --samples-root="${OUTPUT_DIR}/birds/na_rfm" \
    --targets="${targets_arg}" \
    --output="${quality_summary}" \
    --device="${DEVICE}" \
    --batch-size="${QUALITY_BATCH_SIZE}" \
    --num-ref="${NUM_REF}"
fi

echo "birds_camera_ready_done=$(date '+%Y-%m-%dT%H:%M:%S%z')"
