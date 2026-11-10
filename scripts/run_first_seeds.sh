#!/bin/bash
# FlowMamba-Early and FlowMamba-Single-N with the common recipe, seeds 20260921-23, all datasets and exits,
# then import of these runs into the experiment layout. A run whose run_log.txt exists is skipped.
cd "$(dirname "$0")/.." || exit 1
PY=${PYTHON:-python}
RUNS=${FLOWMAMBA_OUTPUT:-outputs}/runs
for dataset in iscxvpn ciciot2022 ustctfc2016 cipherspectrum cstnet-tls1.3; do
  for seed in 20260921 20260922 20260923; do
    if [ ! -f "$RUNS/$dataset/flowmamba_early_${dataset}_seed${seed}/run_log.txt" ]; then
      $PY src/run_flowmamba_early.py --dataset $dataset --seed $seed || echo "FAILED flowmamba_early $dataset $seed"
    fi
    for n in 1 2 3 5; do
      if [ ! -f "$RUNS/$dataset/flowmamba_single_n${n}_${dataset}_seed${seed}/run_log.txt" ]; then
        $PY src/run_flowmamba_single_n.py --dataset $dataset --n_packets $n --seed $seed || echo "FAILED single_n$n $dataset $seed"
      fi
    done
  done
done
$PY src/export_first_seeds.py
