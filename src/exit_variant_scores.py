"""Validation and test macro-F1 of the screened variants of the exit layout and loss weighting.

All runs use seed 20260921 on USTC-TFC2016 and CipherSpectrum. The selected checkpoint of each
run is the epoch with the highest mean validation macro-F1, as in training, and its per-exit
validation scores are read from the same line of the training log. Test scores come from the
summary CSVs of run_flowmamba_early.py, run_exit_variants.py and run_branched_exits.py.
Writes OUTPUT_ROOT/results/exit_variant_scores.csv and exit_variant_scores.md.

    python src/exit_variant_scores.py
"""
import re

import pandas as pd

import experiment as L
from paths import OUTPUT_ROOT, RESULTS_OUT

RUNS = {
    "FlowMamba-Early, as selected": "flowmamba_early_{d}_seed20260921",
    "Exit-specific projections": "flowmamba_v2_adapter_{d}_seed20260921",
    "Learned exit weights": "flowmamba_v2_uncertainty_{d}_seed20260921",
    "Both changes": "flowmamba_v2_both_{d}_seed20260921",
    "Two shared and two exit-specific layers": "flowmamba_v3_{d}_seed20260921_s2b2",
}
EXITS = {"ustctfc2016": [1, 2, 3, 5], "cipherspectrum": [3, 5]}


def selected(path):
    rows = []
    for line in open(path):
        m = re.match(r"epoch (\d+) .*?val_mean_f1 ([0-9.]+) (.*)", line)
        if m:
            per = {int(e): float(v) for e, v in re.findall(r"val_f1@(\d+)=([0-9.]+)", m.group(3))}
            rows.append((int(m.group(1)), float(m.group(2)), per))
    top = max(r[1] for r in rows)
    ties = [r for r in rows if r[1] == top]
    return ties[0], len(ties)


def test_scores(name, d):
    if name.startswith("FlowMamba-Early"):
        t = pd.read_csv(f"{RESULTS_OUT}/flowmamba_early_summary.csv")
        t = t[(t.dataset == d) & (t.seed == 20260921)]
    elif name.startswith("Two shared"):
        t = pd.read_csv(f"{RESULTS_OUT}/flowmamba_v3_summary.csv")
        t = t[(t.dataset == d) & (t.seed == 20260921) & (t.n_shared == 2) & (t.n_branch == 2)]
    else:
        v = {"Exit-specific projections": "adapter", "Learned exit weights": "uncertainty", "Both changes": "both"}[name]
        t = pd.read_csv(f"{RESULTS_OUT}/flowmamba_v2_summary.csv")
        t = t[(t.dataset == d) & (t.seed == 20260921) & (t.variant == v)]
    return {int(r.exit_pos): float(r.test_macro_f1) for r in t.itertuples()}


def main():
    rows = []
    for name, pat in RUNS.items():
        for d, exits in EXITS.items():
            (epoch, mean, per), ties = selected(f"{OUTPUT_ROOT}/runs/{d}/{pat.format(d=d)}/run_log.txt")
            test = test_scores(name, d)
            for e in exits:
                rows.append({"model": name, "dataset": L.NAMES[d], "exit": e, "selected_epoch": epoch,
                             "tied_epochs_at_4_decimals": ties, "val_mean_f1": mean, "val_macro_f1": per[e], "test_macro_f1": test[e]})
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS_OUT}/exit_variant_scores.csv", index=False)
    lines = ["# Screened variants with validation and test scores", "",
             "Seed 20260921. Validation scores are those of the selected checkpoint, read from the training logs "
             "(four decimals); test scores are from the summary CSVs of the training scripts.", "",
             "| Model | Dataset | Exit | Validation macro-F1 | Test macro-F1 | Selected epoch | Tied epochs |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['dataset']} | {r['exit']} | {r['val_macro_f1']:.4f} | {r['test_macro_f1']:.4f} | {r['selected_epoch']} | {r['tied_epochs_at_4_decimals']} |")
    piv = df.pivot_table(index="model", columns=["dataset", "exit"], values="val_macro_f1")
    lines += ["", "Validation macro-F1 by model (rows) and dataset-exit (columns):", "", "```", piv.round(4).to_string(), "```"]
    open(f"{RESULTS_OUT}/exit_variant_scores.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
