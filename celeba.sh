#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
MODE="${MODE:-full}"
OUTPUT_DIR="${OUTPUT_DIR:-results/camera_ready_repro}"
COMBOS="${COMBOS:-all}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-42}"

case "${MODE}" in
  smoke)
    NUM_SAMPLES="${NUM_SAMPLES:-4}"
    SKIP_QUALITY_METRICS="${SKIP_QUALITY_METRICS:-1}"
    ;;
  full)
    NUM_SAMPLES="${NUM_SAMPLES:-256}"
    SKIP_QUALITY_METRICS="${SKIP_QUALITY_METRICS:-0}"
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

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    exit 1
  fi
}

combo_attrs() {
  case "$1" in
    female_nonblond) echo "female nonblond" ;;
    female_blond) echo "female blond" ;;
    male_nonblond) echo "male nonblond" ;;
    male_blond) echo "male blond" ;;
    young_female) echo "young female" ;;
    old_female) echo "old female" ;;
    young_male) echo "young male" ;;
    old_male) echo "old male" ;;
    *) echo "Unknown CelebA combo: $1" >&2; exit 1 ;;
  esac
}

filter_name_for_attr() {
  case "$1" in
    male) echo "celeba_male.pt" ;;
    female) echo "celeba_female.pt" ;;
    young) echo "celeba_young.pt" ;;
    old) echo "celeba_old.pt" ;;
    blond|nonblond) echo "celeba_blond_hair.pt" ;;
    *) echo "Unknown CelebA attribute: $1" >&2; exit 1 ;;
  esac
}

concept_name_for_attr() {
  case "$1" in
    male|female) echo "male_vs_female_mid_block_t250.pt" ;;
    young) echo "young_vs_other_mid_block_t250.pt" ;;
    old) echo "old_vs_other_mid_block_t250.pt" ;;
    blond|nonblond) echo "blond_vs_other_mid_block_t250.pt" ;;
    *) echo "Unknown CelebA attribute: $1" >&2; exit 1 ;;
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

if [[ "${COMBOS}" == "all" ]]; then
  combo_list=(female_nonblond female_blond male_nonblond male_blond young_female old_female young_male old_male)
  combos_arg="all"
else
  normalized_combos="${COMBOS//,/ }"
  combo_list=()
  for combo in ${normalized_combos}; do
    combo_attrs "${combo}" >/dev/null
    combo_list+=("${combo}")
  done
  combos_arg="$(join_csv "${combo_list[@]}")"
fi

launcher_dir="${OUTPUT_DIR}/_launcher/celeba"
mkdir -p "${launcher_dir}"
exec > >(tee -a "${launcher_dir}/launcher.log") 2>&1

echo "celeba_camera_ready_start=$(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "mode=${MODE} output_dir=${OUTPUT_DIR} combos=${combos_arg}"
echo "num_samples=${NUM_SAMPLES} seed=${SEED} batch_size=${BATCH_SIZE} skip_quality=${SKIP_QUALITY_METRICS}"

require_file "repro/assets/celeba_filters/celeba_full.pt"
require_file "repro/assets/celeba_filters/labels_with_hair.pt"
if [[ "${SKIP_QUALITY_METRICS}" != "1" ]]; then
  require_dir "data/celeba_hq_256_30k"
fi

for combo in "${combo_list[@]}"; do
  for attr in $(combo_attrs "${combo}"); do
    require_file "repro/assets/celeba_filters/$(filter_name_for_attr "${attr}")"
    require_file "repro/assets/celeba_concepts/$(concept_name_for_attr "${attr}")"
  done
done

wrapper_args=(
  --task celeba
  --method na_rfm
  --mode "${MODE}"
  --combos "${combos_arg}"
  --num-samples "${NUM_SAMPLES}"
  --seed "${SEED}"
  --batch-size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --output-root "${OUTPUT_DIR}"
  --python "${PYTHON_BIN}"
)
if [[ "${SKIP_QUALITY_METRICS}" == "1" ]]; then
  wrapper_args+=(--skip-quality-metrics)
fi

"${PYTHON_BIN}" repro/run_camera_ready.py "${wrapper_args[@]}"

echo "celeba_camera_ready_done=$(date '+%Y-%m-%dT%H:%M:%S%z')"
