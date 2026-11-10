"""Per-flow metadata of each test split, used to form the evaluation subsets.

capture_in_train marks test flows whose capture file also contributes training flows, and
connection_in_train those whose connection, both directions together, occurs in training;
both use capture() and connection() of analyze_capture_overlap.py. Writes
OUTPUT_ROOT/experiments/meta/<dataset>_test.csv.

    python src/test_metadata.py
"""
import json
import os

import pandas as pd

import analyze_capture_overlap as A
import experiment as L


def main():
    os.makedirs(f"{L.OUT}/meta", exist_ok=True)
    for d in L.DATASETS:
        train = json.load(open(f"{L.DATA_ROOT}/{d}/data-train.json"))
        test = json.load(open(f"{L.DATA_ROOT}/{d}/data-test.json"))
        caps = {A.capture(it["pcap_file"]) for it in train}
        conns = {A.connection(it["pcap_file"]) for it in train} - {None}
        rows = []
        for i, it in enumerate(test):
            c = A.connection(it["pcap_file"])
            rows.append({"index": i, "y_true": it["label"], "npkt": min(it["num_packet"], 5),
                         "capture_in_train": int(A.capture(it["pcap_file"]) in caps),
                         "connection_parsed": int(c is not None), "connection_in_train": int(c in conns) if c else 0,
                         "capture": A.capture(it["pcap_file"]), "connection": str(sorted(c[2])) + c[0] if c else ""})
        df = pd.DataFrame(rows)
        df.to_csv(f"{L.OUT}/meta/{d}_test.csv", index=False)
        print(d, len(df), "unseen connection", int(((df.connection_parsed == 1) & (df.connection_in_train == 0)).sum()),
              "unseen capture", int((df.capture_in_train == 0).sum()))


if __name__ == "__main__":
    main()
