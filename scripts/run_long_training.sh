#!/bin/bash
# Training trajectories of FlowMamba-Early over 150 epochs, seed 20260921, USTC-TFC2016 and CipherSpectrum.
cd "$(dirname "$0")/.." || exit 1
PY=${PYTHON:-python}
for dataset in ustctfc2016 cipherspectrum; do
  $PY src/run_long_training.py --dataset $dataset --seed 20260921 --epochs 150
done
