#!/usr/bin/env bash
set -euo pipefail
trap 'log_error "Pipeline failed at line ${LINENO}."' ERR

# Pipeline order for this repo. This script documents the intended workflow.
# It does not download data; see README "Quick Start" for dataset setup.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# Colors
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
DIM="\033[90m"
NC="\033[0m"

get_time() {
  date +"%H:%M:%S"
}

log_error() {
  printf "%b\n" "${DIM}[$(get_time)]${NC} ${RED}❌ $1${NC}" >&2
}
log_info() {
  if [ "${VERBOSITY}" -ge 1 ]; then
    printf "%b\n" "${DIM}[$(get_time)]${NC} ${BLUE}🚀 $1${NC}"
  fi
}

log_step() {
  if [ "${VERBOSITY}" -ge 1 ]; then
    printf "%b\n" "${DIM}[$(get_time)]${NC} ${YELLOW}⏳ $1${NC}"
  fi
}

log_success() {
  if [ "${VERBOSITY}" -ge 1 ]; then
    printf "%b\n" "${DIM}[$(get_time)]${NC} ${GREEN}✅ $1${NC}"
  fi
}

run_cmd() {
  if [ "${VERBOSITY}" -eq 0 ]; then
    "$@" >/dev/null
  else
    "$@"
  fi
}
DATA_DIR="${ROOT_DIR}/data/emg_data"
OUTPUT_FEATURES="${ROOT_DIR}/data/interim/features_emg_data.csv"

FORCE=0
VERBOSITY=1
for arg in "$@"; do
  case "$arg" in
    -q|--quiet) VERBOSITY=0 ;;
    -v|--verbose) VERBOSITY=2 ;;
    -f|--force) FORCE=1 ;;
  esac
done

if [ "${VERBOSITY}" -le 1 ]; then
  export PYTHONWARNINGS="ignore"
fi

NB_LOG_LEVEL="--log-level ERROR"
NB_NO_INPUT="--no-input"
if [ "${VERBOSITY}" -ge 2 ]; then
  NB_LOG_LEVEL=""
  NB_NO_INPUT=""
fi

log_info "Starting Pipeline..."
log_step "Step 1: Feature extraction (produces features_emg_data.csv)"
if [ "${FORCE}" -eq 1 ]; then
  run_cmd python3 "${ROOT_DIR}/src/feature_extraction.py" -i "${DATA_DIR}"
elif [ -f "${OUTPUT_FEATURES}" ]; then
  log_success "Skipping feature extraction: output already exists."
else
  run_cmd python3 "${ROOT_DIR}/src/feature_extraction.py" -i "${DATA_DIR}"
fi
log_success "Step 1 Complete."

log_step "Step 2: EDA (cleans data; outputs processed CSVs)"
run_cmd jupyter nbconvert --execute --to notebook --inplace ${NB_LOG_LEVEL} "${ROOT_DIR}/notebooks/eda.ipynb"
log_success "Step 2 Complete."

log_step "Step 3: Feature selection compute (produces results/tables/feature_selection.csv)"
run_cmd python3 "${ROOT_DIR}/src/run_feature_selection.py"
log_success "Step 3 Complete."

log_step "Step 4: Modeling compute (produces results/tables/model_comparison.csv)"
run_cmd python3 "${ROOT_DIR}/src/run_modeling.py"
log_success "Step 4 Complete."

log_step "Step 5: Feature selection report notebook"
run_cmd jupyter nbconvert --execute --to notebook --inplace ${NB_LOG_LEVEL} "${ROOT_DIR}/notebooks/feature_selection.ipynb"
log_success "Step 5 Complete."

log_step "Step 6: Modeling report notebook"
run_cmd jupyter nbconvert --execute --to notebook --inplace ${NB_LOG_LEVEL} "${ROOT_DIR}/notebooks/modeling_experiments.ipynb"
log_success "Step 6 Complete."

log_step "Step 7: Final results (analysis and plots)"
run_cmd jupyter nbconvert --execute --to notebook --inplace ${NB_LOG_LEVEL} "${ROOT_DIR}/notebooks/final_results.ipynb"
log_success "Step 7 Complete."
log_step "Step 8: Export final results HTML to reports/"
run_cmd jupyter nbconvert --execute --to html ${NB_NO_INPUT} --output-dir "${ROOT_DIR}/reports" ${NB_LOG_LEVEL} "${ROOT_DIR}/notebooks/final_results.ipynb"
log_success "Step 8 Complete."

log_success "Pipeline complete."
