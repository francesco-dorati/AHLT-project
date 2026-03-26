#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # code/1.1.NERC-ML
cd "$ROOT_DIR"

# Activate venv at repo root
# shellcheck disable=SC1091
source ../../.venv/bin/activate

mkdir -p results_c_change/logs

# Feature extraction is slow; for hyperparameter sweeps we normally reuse existing
# preprocessed/*.feat. Set AHLT_FORCE_EXTRACT=1 to re-extract.
if [[ "${AHLT_FORCE_EXTRACT:-0}" == "1" ]] || [[ ! -s preprocessed/train.feat ]] || [[ ! -s preprocessed/devel.feat ]]; then
  echo "[prep] Extracting features (train + devel)..."
  python -u bin/run.py extract > "results_c_change/logs/extract-$(date +%F_%H%M%S).log" 2>&1
else
  echo "[prep] Skipping extract (using existing preprocessed/*.feat)"
fi

snap() {
  local label="$1"; local model="$2"; shift 2
  local params=("$@"); local params_str
  params_str="${params[*]}"

  echo "[run] $label"
  python -u bin/run.py train predict "$model" "${params[@]}" > "results_c_change/logs/${label}.log" 2>&1

  cp "results/devel-${model}.stats" "results_c_change/${label}.stats"
  cp "results/devel-${model}.out" "results_c_change/${label}.out"

  python bin/append_experiment_summary.py "$label" "$model" "$params_str"
}

# CRF sweeps (c1/c2)
snap devel-CRF-c1_0.1-c2_0.5-it50 CRF c1=0.1 c2=0.5 max_iterations=50
snap devel-CRF-c1_0.05-c2_0.1-it100 CRF c1=0.05 c2=0.1 max_iterations=100
snap devel-CRF-c1_0.2-c2_0.7-it100 CRF c1=0.2 c2=0.7 max_iterations=100
snap devel-CRF-c1_0.1-c2_1-it100 CRF c1=0.1 c2=1 max_iterations=100
snap devel-CRF-c1_0.05-c2_0.5-it100 CRF c1=0.05 c2=0.5 max_iterations=100
snap devel-CRF-c1_0.2-c2_1-it100 CRF c1=0.2 c2=1 max_iterations=100

# MEM sweeps (C)
snap devel-MEM-C1-lbfgs-1500 MEM C=1 solver=lbfgs max_iter=1500
snap devel-MEM-C2-lbfgs-2000 MEM C=2 solver=lbfgs max_iter=2000
snap devel-MEM-C0.5-lbfgs-2000 MEM C=0.5 solver=lbfgs max_iter=2000
snap devel-MEM-C1-lbfgs-2000 MEM C=1 solver=lbfgs max_iter=2000
snap devel-MEM-C5-lbfgs-2000 MEM C=5 solver=lbfgs max_iter=2000

# SVM sweeps (C)
snap devel-SVM-C0.5-rbf SVM C=0.5 kernel=rbf
snap devel-SVM-C1-rbf SVM C=1 kernel=rbf
snap devel-SVM-C5-rbf SVM C=5 kernel=rbf

echo "Done. See results_c_change/experiment-summary.txt"