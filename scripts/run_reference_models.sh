#!/bin/bash
# Reference models trained separately for each packet count N: NetMamba-N, MNB, XGBoost and random forest,
# seeds 20260921-23. A run whose run_log.txt exists is skipped.
cd "$(dirname "$0")/.." || exit 1
PY=${PYTHON:-python}
RUNS=${FLOWMAMBA_OUTPUT:-outputs}/runs
for dataset in iscxvpn ciciot2022 ustctfc2016 cipherspectrum cstnet-tls1.3; do
  for n in 1 2 3 5; do
    for seed in 20260921 20260922 20260923; do
      [ -f "$RUNS/$dataset/netmamba_n${n}_${dataset}_seed${seed}/run_log.txt" ] ||
        $PY src/run_netmamba_n_baseline.py --dataset $dataset --n_packets $n --seed $seed || echo "FAILED netmamba_n$n $dataset $seed"
      [ -f "$RUNS/$dataset/mambanetburst_n${n}_${dataset}_seed${seed}/run_log.txt" ] ||
        $PY src/run_mambanetburst_baseline.py --dataset $dataset --n_packets $n --seed $seed || echo "FAILED mambanetburst_n$n $dataset $seed"
      [ -f "$RUNS/$dataset/xgbrf_n${n}_${dataset}_seed${seed}/run_log.txt" ] ||
        $PY src/run_xgb_rf_baseline.py --dataset $dataset --n_packets $n --seed $seed || echo "FAILED xgb_rf_n$n $dataset $seed"
    done
  done
done
