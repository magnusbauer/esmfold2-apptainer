#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${ESMFOLD2_IMAGE:-${ROOT_DIR}/esmfold2.sif}"
APPTAINER_FLAGS="${ESMFOLD2_APPTAINER_FLAGS:---nv}"
INPUT="${ESMFOLD2_GPU_TEST_INPUT:-/opt/esmfold2/examples/af3_inputs/hf_hhai_1mht.json}"
OUTPUT_ROOT="${ESMFOLD2_GPU_TEST_OUTPUT_DIR:-${ROOT_DIR}/examples/gpu_test_outputs}"
HOST_MODEL_DIR="${ROOT_DIR}/models/ESMFold2"
HOST_CCD_CACHE="${ROOT_DIR}/models/ESMFold2"
MODEL="${ESMFOLD2_GPU_TEST_MODEL:-${ESMFOLD2_MODEL_PATH:-}}"
CCD_CACHE="${ESMFOLD2_GPU_TEST_CCD_CACHE:-${ESMFOLD2_CCD_CACHE:-${ESMFOLD2_CCD_PATH:-}}}"
TIME_LIMIT="${ESMFOLD2_GPU_TEST_TIME:-02:00:00}"
MEMORY="${ESMFOLD2_GPU_TEST_MEM:-64g}"
CPUS="${ESMFOLD2_GPU_TEST_CPUS:-4}"
SEED="${ESMFOLD2_GPU_TEST_SEED:-0}"
NUM_LOOPS="${ESMFOLD2_GPU_TEST_NUM_LOOPS:-3}"
NUM_SAMPLING_STEPS="${ESMFOLD2_GPU_TEST_NUM_SAMPLING_STEPS:-200}"
NUM_DIFFUSION_SAMPLES="${ESMFOLD2_GPU_TEST_NUM_DIFFUSION_SAMPLES:-1}"
USE_MODEL_DEFAULTS="${ESMFOLD2_GPU_TEST_USE_MODEL_DEFAULTS:-1}"

if [[ -z "${MODEL}" && -d "${HOST_MODEL_DIR}" ]]; then
  MODEL="${HOST_MODEL_DIR}"
fi
if [[ -z "${CCD_CACHE}" && -d "${HOST_CCD_CACHE}" ]]; then
  CCD_CACHE="${HOST_CCD_CACHE}"
fi

CASES=(
  "ada_4000|gpu|4000Ada|small"
  "blackwell_b4000|gpu|B4000|small"
  "blackwell_b6000|gpu|B6000|large"
  "blackwell_b6000q|gpu|B6000Q|large"
  "l40|gpu-bf|L40|large"
  "l40s|gpu-bf|L40S|large"
  "a100|gpu-bf|A100|large"
  "a6000|gpu|A6000|large"
)

usage() {
  cat <<'EOF'
Usage:
  examples/run_gpu_arch_tests.sh [submit-case ...]
  examples/run_gpu_arch_tests.sh --run-case CASE
  examples/run_gpu_arch_tests.sh --list-cases

Submits one Slurm job per GPU architecture case and leaves output folders under
examples/gpu_test_outputs by default. Each successful run writes:
  af3_outputs/<job_id>/<job_id>_model.cif
  af3_outputs/<job_id>/<job_id>_confidences.json
  af3_outputs/<job_id>/<job_id>_full_metrics.pkl
  af3_outputs/<job_id>/<job_id>_ranking_scores.csv
  metrics.json
  command.txt
  gpu_info.txt

Environment overrides:
  ESMFOLD2_IMAGE=/path/to/esmfold2.sif
  ESMFOLD2_APPTAINER_FLAGS='--nv'
  ESMFOLD2_GPU_TEST_MODEL=/path/to/models/ESMFold2
  ESMFOLD2_GPU_TEST_CCD_CACHE=/path/to/models/ESMFold2
  ESMFOLD2_GPU_TEST_OUTPUT_DIR=/path/to/output_root
  ESMFOLD2_GPU_TEST_INPUT=/path/or/container/path/input.json
  ESMFOLD2_GPU_TEST_USE_MODEL_DEFAULTS=1   # default; omit loop/step flags
  ESMFOLD2_GPU_TEST_USE_MODEL_DEFAULTS=0   # quick CUDA smoke only
  ESMFOLD2_GPU_TEST_NUM_LOOPS=3
  ESMFOLD2_GPU_TEST_NUM_SAMPLING_STEPS=200
  ESMFOLD2_GPU_TEST_NUM_DIFFUSION_SAMPLES=1
EOF
}

case_spec() {
  local wanted="$1"
  local spec name
  for spec in "${CASES[@]}"; do
    IFS='|' read -r name _ _ _ <<<"${spec}"
    if [[ "${name}" == "${wanted}" ]]; then
      printf '%s\n' "${spec}"
      return 0
    fi
  done
  return 1
}

list_cases() {
  local spec name partition constraint gres
  for spec in "${CASES[@]}"; do
    IFS='|' read -r name partition constraint gres <<<"${spec}"
    printf '%-18s partition=%-7s constraint=%-10s gres=gpu:%s:1\n' \
      "${name}" "${partition}" "${constraint}" "${gres}"
  done
}

run_case() {
  local case_name="$1"
  local spec partition constraint gres
  spec="$(case_spec "${case_name}")" || {
    echo "Unknown case: ${case_name}" >&2
    list_cases >&2
    exit 2
  }
  IFS='|' read -r _ partition constraint gres <<<"${spec}"

  local out_dir="${OUTPUT_ROOT}/${case_name}"
  mkdir -p "${out_dir}"
  printf '%s\n' "${IMAGE}" >"${out_dir}/image_source.txt"

  {
    date -Is
    hostname
    nvidia-smi -L
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
  } >"${out_dir}/gpu_info.txt"

  local cmd=(
    apptainer exec
  )
  # shellcheck disable=SC2206
  local apptainer_flags=( ${APPTAINER_FLAGS} )
  cmd+=(
    "${apptainer_flags[@]}"
  )
  if [[ -n "${MODEL}" ]]; then
    cmd+=(--env "ESMFOLD2_MODEL_PATH=${MODEL}")
  fi
  if [[ -n "${CCD_CACHE}" ]]; then
    cmd+=(
      --env "ESMFOLD2_CCD_CACHE=${CCD_CACHE}"
      --env "ESMFOLD2_CCD_PATH=${CCD_CACHE}/ccd.pkl"
      --env "ESMCFOLD_CCD_PATH=${CCD_CACHE}/ccd.pkl"
    )
  fi
  cmd+=(
    "${IMAGE}" esmfold2_predict
    --input "${INPUT}"
    --output-dir "${out_dir}/af3_outputs"
    --metrics-json "${out_dir}/metrics.json"
    --full-metrics
    --seed "${SEED}"
  )
  if [[ -n "${MODEL}" ]]; then
    cmd+=(--model "${MODEL}")
  fi
  if [[ -n "${CCD_CACHE}" ]]; then
    cmd+=(--ccd-cache "${CCD_CACHE}")
  fi

  if [[ "${USE_MODEL_DEFAULTS}" != "1" ]]; then
    cmd+=(
      --num-loops "${NUM_LOOPS}"
      --num-sampling-steps "${NUM_SAMPLING_STEPS}"
      --num-diffusion-samples "${NUM_DIFFUSION_SAMPLES}"
    )
  fi

  printf '%q ' "${cmd[@]}" >"${out_dir}/command.txt"
  printf '\n' >>"${out_dir}/command.txt"

  local start_epoch end_epoch status
  start_epoch="$(date +%s)"
  {
    printf 'case=%s\n' "${case_name}"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-}"
    printf 'node=%s\n' "$(hostname)"
    printf 'started_at=%s\n' "$(date -Is)"
    printf 'start_epoch=%s\n' "${start_epoch}"
  } >"${out_dir}/timing.txt"

  set +e
  "${cmd[@]}" 2>&1 | tee "${out_dir}/run.log"
  status="${PIPESTATUS[0]}"
  set -e

  end_epoch="$(date +%s)"
  {
    printf 'finished_at=%s\n' "$(date -Is)"
    printf 'end_epoch=%s\n' "${end_epoch}"
    printf 'elapsed_seconds=%s\n' "$((end_epoch - start_epoch))"
    printf 'exit_code=%s\n' "${status}"
  } >>"${out_dir}/timing.txt"
  return "${status}"
}

submit_case() {
  local case_name="$1"
  local spec partition constraint gres
  spec="$(case_spec "${case_name}")" || {
    echo "Unknown case: ${case_name}" >&2
    list_cases >&2
    exit 2
  }
  IFS='|' read -r _ partition constraint gres <<<"${spec}"

  local out_dir="${OUTPUT_ROOT}/${case_name}"
  local script_path
  script_path="$(readlink -f "${BASH_SOURCE[0]}")"
  mkdir -p "${out_dir}"
  sbatch \
    --partition="${partition}" \
    --constraint="${constraint}" \
    --gres="gpu:${gres}:1" \
    --mem="${MEMORY}" \
    --cpus-per-task="${CPUS}" \
    --time="${TIME_LIMIT}" \
    --job-name="ef2_${case_name}" \
    --output="${out_dir}/slurm-%j.out" \
    --export=ALL,ESMFOLD2_IMAGE="${IMAGE}",ESMFOLD2_APPTAINER_FLAGS="${APPTAINER_FLAGS}",ESMFOLD2_GPU_TEST_INPUT="${INPUT}",ESMFOLD2_GPU_TEST_OUTPUT_DIR="${OUTPUT_ROOT}",ESMFOLD2_GPU_TEST_MODEL="${MODEL}",ESMFOLD2_GPU_TEST_CCD_CACHE="${CCD_CACHE}",ESMFOLD2_GPU_TEST_USE_MODEL_DEFAULTS="${USE_MODEL_DEFAULTS}",ESMFOLD2_GPU_TEST_NUM_LOOPS="${NUM_LOOPS}",ESMFOLD2_GPU_TEST_NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS}",ESMFOLD2_GPU_TEST_NUM_DIFFUSION_SAMPLES="${NUM_DIFFUSION_SAMPLES}",ESMFOLD2_GPU_TEST_SEED="${SEED}" \
    --wrap="bash ${script_path} --run-case ${case_name}"
}

main() {
  if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
  fi
  if [[ "${1:-}" == "--list-cases" ]]; then
    list_cases
    exit 0
  fi
  if [[ "${1:-}" == "--run-case" ]]; then
    if [[ -z "${2:-}" ]]; then
      echo "--run-case requires a case name" >&2
      exit 2
    fi
    run_case "$2"
    exit 0
  fi

  local selected=("$@")
  if [[ "${#selected[@]}" -eq 0 ]]; then
    local spec name
    selected=()
    for spec in "${CASES[@]}"; do
      IFS='|' read -r name _ _ _ <<<"${spec}"
      selected+=("${name}")
    done
  fi

  local case_name
  for case_name in "${selected[@]}"; do
    submit_case "${case_name}"
  done
}

main "$@"
