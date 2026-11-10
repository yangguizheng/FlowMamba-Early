"""XGBoost and random-forest baselines on the packet sizes and inter-arrival times of the first N packets.

The features of a flow with at least N packets are the first N values of the sizes and
intervals fields of the processed release. Test scores of both models are appended to
OUTPUT_ROOT/results/xgb_rf_summary.csv:

    python src/run_xgb_rf_baseline.py --dataset iscxvpn --n_packets 3 --seed 20260921
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from paths import DATA_ROOT, OUTPUT_ROOT, RESULTS_OUT


def parse_floats(s, n):
    vals = [float(x) for x in s.split()][:n]
    vals += [0.0] * (n - len(vals))
    return vals


def load_split(dataset, split, n_packets):
    with open(f"{DATA_ROOT}/{dataset}/{split}.json") as f:
        raw = json.load(f)
    raw = [item for item in raw if item["num_packet"] >= n_packets]
    X, y = [], []
    for item in raw:
        sizes = parse_floats(item["sizes"], n_packets)
        intervals = parse_floats(item["intervals"], n_packets)
        X.append(sizes + intervals)
        y.append(item["label"])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), len(raw)


def get_nb_classes(dataset):
    with open(f"{DATA_ROOT}/{dataset}/metadata.json") as f:
        meta = json.load(f)
    return max(meta["name_to_idx"].values()) + 1


def evaluate(model, X, y):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1m = f1_score(y, preds, average="macro", zero_division=0)
    try:
        mcc = matthews_corrcoef(y, preds)
    except ValueError:
        mcc = float("nan")
    return acc, f1m, mcc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n_packets", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()

    nb_classes = get_nb_classes(args.dataset)
    X_train, y_train, n_train = load_split(args.dataset, "data-train", args.n_packets)
    X_valid, y_valid, n_valid = load_split(args.dataset, "data-valid", args.n_packets)
    X_test, y_test, n_test = load_split(args.dataset, "data-test", args.n_packets)

    # At larger N, some rare classes may have zero eligible (num_packet >= N) samples in a
    # given split; XGBoost's multi:softmax with an explicit num_class requires labels to be a
    # dense 0..k-1 range, so remap to the classes actually present in this eligible subset.
    encoder = LabelEncoder()
    encoder.fit(np.concatenate([y_train, y_valid, y_test]))
    n_classes_present = len(encoder.classes_)
    y_train = encoder.transform(y_train)
    y_valid = encoder.transform(y_valid)
    y_test = encoder.transform(y_test)

    run_id = f"xgbrf_n{args.n_packets}_{args.dataset}_seed{args.seed}"
    run_dir = f"{OUTPUT_ROOT}/runs/{args.dataset}/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    log_lines = []
    results = {}
    for name in ["xgb", "rf"]:
        t0 = time.time()
        if name == "xgb":
            model = XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                objective="multi:softmax", num_class=n_classes_present,
                random_state=args.seed, n_jobs=8, eval_metric="mlogloss",
            )
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        else:
            model = RandomForestClassifier(
                n_estimators=300, max_depth=None, random_state=args.seed, n_jobs=8,
            )
            model.fit(X_train, y_train)
        elapsed = time.time() - t0
        val_acc, val_f1, val_mcc = evaluate(model, X_valid, y_valid)
        test_acc, test_f1, test_mcc = evaluate(model, X_test, y_test)
        line = (f"{name} n_packets={args.n_packets} val_f1={val_f1:.4f} "
                f"test_acc={test_acc:.4f} test_f1={test_f1:.4f} test_mcc={test_mcc:.4f} elapsed={elapsed:.1f}s")
        print(line, flush=True)
        log_lines.append(line)
        results[name] = {
            "val_acc": val_acc, "val_macro_f1": val_f1, "val_mcc": val_mcc,
            "test_acc": test_acc, "test_macro_f1": test_f1, "test_mcc": test_mcc,
            "elapsed_sec": elapsed,
        }

    result = {
        "dataset": args.dataset, "n_packets": args.n_packets, "seed": args.seed, "nb_classes": nb_classes,
        "n_classes_present": n_classes_present,
        "n_train": n_train, "n_valid": n_valid, "n_test": n_test,
        "xgb": results["xgb"], "rf": results["rf"],
    }
    with open(f"{run_dir}/run_log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n" + json.dumps(result, indent=2))
    print("RESULT_JSON " + json.dumps(result), flush=True)

    os.makedirs(RESULTS_OUT, exist_ok=True)
    summary_path = f"{RESULTS_OUT}/xgb_rf_summary.csv"
    write_header = not os.path.exists(summary_path)
    cols = ["dataset", "n_packets", "seed", "nb_classes", "n_classes_present", "model", "n_train", "n_valid", "n_test",
            "val_macro_f1", "test_acc", "test_macro_f1", "test_mcc", "elapsed_sec"]
    with open(summary_path, "a") as f:
        if write_header:
            f.write(",".join(cols) + "\n")
        for name in ["xgb", "rf"]:
            row = {
                "dataset": args.dataset, "n_packets": args.n_packets, "seed": args.seed, "nb_classes": nb_classes,
                "n_classes_present": n_classes_present,
                "model": name, "n_train": n_train, "n_valid": n_valid, "n_test": n_test,
                "val_macro_f1": results[name]["val_macro_f1"], "test_acc": results[name]["test_acc"],
                "test_macro_f1": results[name]["test_macro_f1"], "test_mcc": results[name]["test_mcc"],
                "elapsed_sec": results[name]["elapsed_sec"],
            }
            f.write(",".join(str(row[c]) for c in cols) + "\n")


if __name__ == "__main__":
    main()
