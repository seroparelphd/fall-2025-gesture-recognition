#!/usr/bin/env bash
set -euo pipefail

# Pipeline order for this repo. This script documents the intended workflow.
# It does not download data; see README "Quick Start" for dataset setup.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "1) Feature extraction (produces features_emg_data.csv)"
python3 "${ROOT_DIR}/src/feature_extraction.py" -i "${HOME}/emg_data"

echo "2) EDA (cleans data; outputs processed CSVs)"
jupyter nbconvert --execute --to notebook --inplace "${ROOT_DIR}/notebooks/eda.ipynb"

echo "3) Feature selection (produces results/feature_selection.csv and processed train/test splits)"
jupyter nbconvert --execute --to notebook --inplace "${ROOT_DIR}/notebooks/feature_selection.ipynb"

echo "4) Modeling experiments (produces results/model_comparison.csv)"
jupyter nbconvert --execute --to notebook --inplace "${ROOT_DIR}/notebooks/modeling_experiments.ipynb"

echo "5) Final results (analysis and plots)"
jupyter nbconvert --execute --to notebook --inplace "${ROOT_DIR}/notebooks/final_results.ipynb"
echo "6) Export final results HTML to reports/"
jupyter nbconvert --execute --to html --output-dir "${ROOT_DIR}/reports" "${ROOT_DIR}/notebooks/final_results.ipynb"

echo "Pipeline complete."
