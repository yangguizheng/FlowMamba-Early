#!/bin/bash
# Screened variants of the exit layout and loss weighting, seed 20260921, USTC-TFC2016 and CipherSpectrum,
# then their validation and test scores.
cd "$(dirname "$0")/.." || exit 1
PY=${PYTHON:-python}
for dataset in ustctfc2016 cipherspectrum; do
  for variant in adapter uncertainty both; do
    $PY src/run_exit_variants.py --dataset $dataset --seed 20260921 --variant $variant
  done
  $PY src/run_branched_exits.py --dataset $dataset --seed 20260921 --n_shared 2 --n_branch 2
done
$PY src/exit_variant_scores.py
