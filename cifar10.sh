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
BATCH_SIZE="${BATCH_SIZE:-64}"
NA_ONLY_BATCH_SIZE="${NA_ONLY_BATCH_SIZE:-64}"
QUALITY_BATCH_SIZE="${QUALITY_BATCH_SIZE:-64}"
NUM_REF="${NUM_REF:--1}"
RUN_NA_ONLY="${RUN_NA_ONLY:-1}"
RUN_NA_RFM="${RUN_NA_RFM:-1}"

case "${MODE}" in
  smoke)
    NUM_SAMPLES="${NUM_SAMPLES:-4}"
    RUN_QUALITY="${RUN_QUALITY:-0}"
    ;;
  full)
    NUM_SAMPLES="${NUM_SAMPLES:-2048}"
    RUN_QUALITY="${RUN_QUALITY:-1}"
    ;;
  *)
    echo "MODE must be smoke or full, got: ${MODE}" >&2
    exit 1
    ;;
esac

require_file() {
  if [[ ! -e "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

cifar_class_name() {
  case "$1" in
    0|airplane) echo "airplane" ;;
    1|automobile) echo "automobile" ;;
    2|bird) echo "bird" ;;
    3|cat) echo "cat" ;;
    4|deer) echo "deer" ;;
    5|dog) echo "dog" ;;
    6|frog) echo "frog" ;;
    7|horse) echo "horse" ;;
    8|ship) echo "ship" ;;
    9|truck) echo "truck" ;;
    *) echo "Unknown CIFAR-10 target: $1" >&2; exit 1 ;;
  esac
}

cifar_class_idx() {
  case "$1" in
    0|airplane) echo "0" ;;
    1|automobile) echo "1" ;;
    2|bird) echo "2" ;;
    3|cat) echo "3" ;;
    4|deer) echo "4" ;;
    5|dog) echo "5" ;;
    6|frog) echo "6" ;;
    7|horse) echo "7" ;;
    8|ship) echo "8" ;;
    9|truck) echo "9" ;;
    *) echo "Unknown CIFAR-10 target: $1" >&2; exit 1 ;;
  esac
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
  target_list=(0 1 2 3 4 5 6 7 8 9)
  targets_arg="all"
else
  normalized_targets="${TARGETS//,/ }"
  target_list=()
  for target in ${normalized_targets}; do
    target_list+=("$(cifar_class_idx "${target}")")
  done
  targets_arg="$(join_csv "${target_list[@]}")"
fi

launcher_dir="${OUTPUT_DIR}/_launcher/cifar10"
mkdir -p "${launcher_dir}"
exec > >(tee -a "${launcher_dir}/launcher.log") 2>&1

echo "cifar10_camera_ready_start=$(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "mode=${MODE} output_dir=${OUTPUT_DIR} targets=${targets_arg}"
echo "num_samples=${NUM_SAMPLES} seed_start=${SEED_START} batch_size=${BATCH_SIZE}"

require_file "data/checkpoints/openai_cifar10.pt"

if [[ "${RUN_NA_ONLY}" == "1" ]]; then
  require_file "repro/assets/cifar10_filters/full_dataset.pt"
  for target in "${target_list[@]}"; do
    require_file "repro/assets/cifar10_filters/class_${target}.pt"
  done
fi

if [[ "${RUN_NA_RFM}" == "1" ]]; then
  require_file "repro/assets/cifar10_filters/full_dataset.pt"
  for target in "${target_list[@]}"; do
    class_name="$(cifar_class_name "${target}")"
    require_file "repro/assets/cifar10_filters/class_${target}.pt"
    if [[ ! -e "repro/assets/cifar10_concepts/${class_name}.pt" &&
          ! -e "repro/assets/cifar10_concepts/class_${target}.pt" &&
          ! -e "repro/assets/cifar10_concepts/binary_rfm_${class_name}_t93_input_blocks_9.pt" ]]; then
      echo "Missing CIFAR-10 RFM concept for ${target}/${class_name} under repro/assets/cifar10_concepts" >&2
      exit 1
    fi
  done
fi

if [[ "${RUN_NA_ONLY}" == "1" ]]; then
  "${PYTHON_BIN}" repro/run_camera_ready.py \
    --task cifar10 \
    --method na_only \
    --mode "${MODE}" \
    --targets "${targets_arg}" \
    --num-samples "${NUM_SAMPLES}" \
    --batch-size "${NA_ONLY_BATCH_SIZE}" \
    --device "${DEVICE}" \
    --output-root "${OUTPUT_DIR}" \
    --python "${PYTHON_BIN}"
fi

if [[ "${RUN_NA_RFM}" == "1" ]]; then
  "${PYTHON_BIN}" repro/run_camera_ready.py \
    --task cifar10 \
    --method na_rfm \
    --mode "${MODE}" \
    --targets "${targets_arg}" \
    --num-samples "${NUM_SAMPLES}" \
    --seed-start "${SEED_START}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --output-root "${OUTPUT_DIR}" \
    --python "${PYTHON_BIN}"
fi

if [[ "${RUN_QUALITY}" == "1" && "${RUN_NA_RFM}" == "1" ]]; then
  quality_summary="${OUTPUT_DIR}/cifar10/na_rfm/quality_summary.json"
  "${PYTHON_BIN}" repro/evaluate_saved_samples.py \
    --dataset=cifar10 \
    --samples-root="${OUTPUT_DIR}/cifar10/na_rfm" \
    --targets="${targets_arg}" \
    --output="${quality_summary}" \
    --device="${DEVICE}" \
    --batch-size="${QUALITY_BATCH_SIZE}" \
    --num-ref="${NUM_REF}"
fi

echo "cifar10_camera_ready_done=$(date '+%Y-%m-%dT%H:%M:%S%z')"
