"""Import the runs of seeds 20260921-23 of run_flowmamba_early.py and run_flowmamba_single_n.py
into the experiment layout of experiment.py.

For every dataset, seed and model, the selected checkpoint is evaluated on the validation and
test splits without a mask (variant orig) and with each mask of masks.py applied only to the
inputs (variants orig_at_mask_sni, orig_at_mask_hdr, orig_at_mask_both, orig_at_mask_trailer
and orig_at_mask_all). Inference is deterministic, so the unmasked test macro-F1 must equal
the summary CSVs written by the training scripts; any difference is listed and the script
exits with a non-zero status.

    python src/export_first_seeds.py
"""
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import experiment as L
from paths import OUTPUT_ROOT, RESULTS_OUT

SEEDS = [20260921, 20260922, 20260923]
MASKS = [None, "mask_sni", "mask_hdr", "mask_both", "mask_trailer", "mask_all"]


def best_epoch_from_log(path, key):
    vals = []
    for line in open(path):
        m = re.search(key + r" ([0-9.]+)", line) if line.startswith("epoch") else None
        if m:
            vals.append(float(m.group(1)))
    top = max(vals)
    cands = [i for i, v in enumerate(vals) if v == top]
    return (cands[0] if len(cands) == 1 else None), cands


def main():
    fme_csv = pd.read_csv(f"{RESULTS_OUT}/flowmamba_early_summary.csv")
    sgl_csv = pd.read_csv(f"{RESULTS_OUT}/flowmamba_single_n_summary.csv")
    mismatches = []
    for d in L.DATASETS:
        nb = L.orig.get_nb_classes(d)
        n_train = len(json.load(open(f"{L.DATA_ROOT}/{d}/data-train.json")))
        for mask in MASKS:
            variant = "orig" if mask is None else f"orig_at_{mask}"
            split = {s: L.load_split(d, s, mask) for s in ["valid", "test"]}
            for seed in SEEDS:
                for kind, n in [("FME", None)] + [("Single", k) for k in L.EXITS]:
                    job = {"model": kind, "variant": variant, "dataset": d, "seed": seed, "n": n, "budget": "orig", "mask": mask}
                    run_dir, tag = L.run_dir_for(job)
                    if os.path.exists(f"{run_dir}/result.json"):
                        continue
                    os.makedirs(run_dir, exist_ok=True)
                    src = (f"{OUTPUT_ROOT}/runs/{d}/flowmamba_early_{d}_seed{seed}" if kind == "FME"
                           else f"{OUTPUT_ROOT}/runs/{d}/flowmamba_single_n{n}_{d}_seed{seed}")
                    exits = L.EXITS if kind == "FME" else [n]
                    model = L.orig.FlowMambaEarly(num_classes=nb, exit_positions=exits, device="cuda").cuda()
                    model.load_state_dict(torch.load(f"{src}/checkpoint-best.pth", map_location="cuda", weights_only=True))
                    sets = {s: (L.MultiExitSet(split[s]) if kind == "FME" else L.TruncatedSet(split[s], n)) for s in split}
                    arrays, test = {}, None
                    for s in ["valid", "test"]:
                        pred = L.predict(model, DataLoader(sets[s], batch_size=128, shuffle=False, num_workers=2), exits, kind == "FME")
                        for e, v in pred.items():
                            arrays[f"{s}_e{e}_index"] = sets[s].index[v["pos"]]
                            arrays[f"{s}_e{e}_y"] = v["y"].astype(np.int16)
                            arrays[f"{s}_e{e}_pred"] = v["p"].astype(np.int16)
                            arrays[f"{s}_e{e}_prob"] = v["prob"].astype(np.float32)
                            arrays[f"{s}_e{e}_npkt"] = sets[s].npkt[v["pos"]].astype(np.int8)
                        if s == "test":
                            test = L.scores(pred)
                    np.savez_compressed(f"{run_dir}/preds_best.npz", **arrays)
                    if kind == "FME":
                        rows = fme_csv[(fme_csv.dataset == d) & (fme_csv.seed == seed)]
                        epochs = int(rows.epochs_ran.iloc[0])
                        steps = epochs * math.ceil(n_train / 128)
                        be, cands = best_epoch_from_log(f"{src}/run_log.txt", "val_mean_f1")
                        ref = {e: float(rows[rows.exit_pos == e].test_macro_f1.iloc[0]) for e in exits}
                        elapsed = float(rows.elapsed_sec.iloc[0])
                    else:
                        row = sgl_csv[(sgl_csv.dataset == d) & (sgl_csv.seed == seed) & (sgl_csv.n_packets == n)].iloc[0]
                        epochs, steps, elapsed = int(row.epochs_ran), int(row.n_updates), float(row.elapsed_sec)
                        be, cands = best_epoch_from_log(f"{src}/run_log.txt", "val_f1")
                        ref = {n: float(row.test_macro_f1)}
                    if mask is None:
                        for e in exits:
                            if abs(test[e]["f1"] - ref[e]) > 1e-12:
                                mismatches.append((tag, e, test[e]["f1"], ref[e]))
                    config = {**job, "run_id": tag, "source_run": src, "source_checkpoint_sha256": L.sha256(f"{src}/checkpoint-best.pth"),
                              "exits": exits, "epochs_run": epochs, "best_epoch": be, "best_epoch_log_candidates": cands,
                              "best_epoch_note": "zero-based; the training logs round validation scores to four decimals",
                              "optimizer_steps": steps, "optimizer_steps_note": "FME: epochs x ceil(train/128), computed; Single: logged n_updates",
                              "train_flows": n_train if kind == "FME" else int(row.n_train), "wall_time_min": elapsed / 60,
                              "status": "ok", **L.env_record()}
                    json.dump(config, open(f"{run_dir}/config.json", "w"), indent=2)
                    json.dump({"run_id": tag, "test": {str(e): test[e] for e in exits}, "summary_csv_test_f1": {str(e): ref[e] for e in exits}},
                              open(f"{run_dir}/result.json", "w"), indent=2)
                    print(f"{tag} " + " ".join(f"e{e}={test[e]['f1']:.4f}" for e in exits), flush=True)
    if mismatches:
        print("MISMATCHES against the summary CSVs:")
        for m in mismatches:
            print(m)
        sys.exit(1)
    print("EXPORT_DONE unmasked test macro-F1 identical to the summary CSVs for all runs")


if __name__ == "__main__":
    main()
