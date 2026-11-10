"""Descriptive statistics of the five processed datasets; no model is run.

For every dataset: the class names and, per split, the number of flows, the number of
flows per class, the distribution of the number of real packets capped at five, and the
largest raw packet count. Writes OUTPUT_ROOT/results/dataset_stats.json.

    python src/dataset_stats.py
"""
import json
import os
from collections import Counter

from paths import DATA_ROOT, RESULTS_OUT

DATASETS = ["iscxvpn", "ciciot2022", "ustctfc2016", "cipherspectrum", "cstnet-tls1.3"]


def main():
    out = {}
    for ds in DATASETS:
        meta = json.load(open(f"{DATA_ROOT}/{ds}/metadata.json"))
        name_to_idx = meta["name_to_idx"]
        rec = {"classes": {str(v): k for k, v in name_to_idx.items()}, "splits": {}}
        for split in ["train", "valid", "test"]:
            raw = json.load(open(f"{DATA_ROOT}/{ds}/data-{split}.json"))
            labels = Counter(item["label"] for item in raw)
            npkt = Counter(min(item["num_packet"], 5) for item in raw)
            rec["splits"][split] = {
                "n": len(raw),
                "per_class": {str(k): v for k, v in sorted(labels.items())},
                "num_packet_capped": {str(k): v for k, v in sorted(npkt.items())},
                "num_packet_max_raw": max(item["num_packet"] for item in raw),
            }
            del raw
            print(ds, split, rec["splits"][split]["n"], flush=True)
        out[ds] = rec
    os.makedirs(RESULTS_OUT, exist_ok=True)
    with open(f"{RESULTS_OUT}/dataset_stats.json", "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
