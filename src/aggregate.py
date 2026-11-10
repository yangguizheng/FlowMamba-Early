"""Collect every finished run of the experiment layout into one table of scores.

Writes OUTPUT_ROOT/results/per_seed.csv, one row per run, exit and evaluation set, and one
parquet file of per-flow predictions per run under OUTPUT_ROOT/experiments/preds/. The test
evaluation sets are all test flows (test), the flows with at least five packets (test_common), the
flows of connections absent from training (test_unseen_conn) and the flows of capture files
absent from training (test_unseen_file); val is the validation split. Macro-F1 averages over
the classes that occur among the true or predicted labels.

    python src/aggregate.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import experiment as L
from paths import RESULTS_OUT

COLS = ["dataset", "model", "variant", "seed", "exit", "eval_set", "n_flows", "n_classes_present", "macro_f1", "accuracy",
        "train_flows", "epochs_run", "best_epoch", "optimizer_steps", "wall_time_min", "status"]


def subsets(meta, index, npkt):
    m = meta.loc[index]
    return {"test": np.ones(len(index), bool), "test_common": npkt >= 5,
            "test_unseen_conn": ((m.connection_parsed == 1) & (m.connection_in_train == 0)).to_numpy(),
            "test_unseen_file": (m.capture_in_train == 0).to_numpy()}


def main():
    metas = {d: pd.read_csv(f"{L.OUT}/meta/{d}_test.csv").set_index("index") for d in L.DATASETS}
    os.makedirs(f"{L.OUT}/preds", exist_ok=True)
    rows = []
    for res in sorted(glob.glob(f"{L.OUT}/runs/*/*/*/result.json")):
        run = os.path.dirname(res)
        cfg = json.load(open(f"{run}/config.json"))
        z = np.load(f"{run}/preds_best.npz")
        base = {"dataset": L.NAMES[cfg["dataset"]], "model": cfg["model"], "variant": cfg["variant"], "seed": cfg["seed"],
                "train_flows": cfg.get("train_flows"), "epochs_run": cfg.get("epochs_run"), "best_epoch": cfg.get("best_epoch"),
                "optimizer_steps": cfg.get("optimizer_steps"), "wall_time_min": cfg.get("wall_time_min"), "status": cfg.get("status", "ok")}
        flows = []
        for e in cfg["exits"]:
            for split in ["valid", "test"]:
                idx, y, p = z[f"{split}_e{e}_index"], z[f"{split}_e{e}_y"], z[f"{split}_e{e}_pred"]
                npkt = z[f"{split}_e{e}_npkt"]
                flows.append(pd.DataFrame({"flow_id": [f"{cfg['dataset']}:{split}:{i}" for i in idx], "split": split, "exit": e,
                                           "n_packets": npkt, "y_true": y, "y_pred": p, "top1_prob": z[f"{split}_e{e}_prob"]}))
                sets = {"val": np.ones(len(idx), bool)} if split == "valid" else subsets(metas[cfg["dataset"]], idx, npkt)
                for name, m in sets.items():
                    ok = m.sum() > 0
                    rows.append({**base, "exit": e, "eval_set": name, "n_flows": int(m.sum()),
                                 "n_classes_present": int(len(np.unique(y[m]))) if ok else 0,
                                 "macro_f1": float(f1_score(y[m], p[m], average="macro", zero_division=0)) if ok else None,
                                 "accuracy": float(accuracy_score(y[m], p[m])) if ok else None})
        tag = f"{cfg['model']}_{cfg['variant']}_{L.NAMES[cfg['dataset']]}" + (f"_n{cfg['n']}" if cfg.get("n") else "") + f"_s{cfg['seed']}"
        pd.concat(flows).to_parquet(f"{L.OUT}/preds/{tag}.parquet", index=False)
    df = pd.DataFrame(rows, columns=COLS)
    os.makedirs(RESULTS_OUT, exist_ok=True)
    df.to_csv(f"{RESULTS_OUT}/per_seed.csv", index=False)
    print(f"{len(df)} rows from {df.groupby(['model', 'variant', 'dataset', 'seed']).ngroups} model-variant-dataset-seed groups")
    print(df.groupby(["model", "variant"]).seed.nunique().to_string())


if __name__ == "__main__":
    main()
