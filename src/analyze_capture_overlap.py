"""Overlap of capture files and connections between the training and test splits.

The samples are unidirectional flows. The pcap_file field of a sample names the original
capture file (the path up to its first ".pcap", including the class directory) and ends
with the flow's own endpoints; CipherSpectrum inserts the bidirectional connection between
the two. A connection is identified by everything before the last flow component plus the
unordered endpoint pair, so both directions of a connection share one key.

For each dataset the script counts the test flows whose capture file or connection also
occurs in training, and the classes whose flows all come from a single capture file.
Writes OUTPUT_ROOT/results/capture_overlap.json. test_metadata.py uses capture() and
connection() to mark the test flows of unseen connections and unseen capture files.

    python src/analyze_capture_overlap.py
"""
import json
import os
import re
from collections import defaultdict

from paths import DATA_ROOT, RESULTS_OUT

DATASETS = ["iscxvpn", "ciciot2022", "ustctfc2016", "cipherspectrum", "cstnet-tls1.3"]
FLOW = re.compile(r"([A-Z]+)_([0-9-]+)_(\d+)_([0-9-]+)_(\d+)\.pcap$")


def capture(path):
    return path[:path.index(".pcap") + 5]


def connection(path):
    head, _, tail = path.rpartition(".pcap.")
    m = FLOW.match(tail)
    if not m:
        return None
    proto, a, pa, b, pb = m.groups()
    return head, proto, frozenset([(a, pa), (b, pb)])


def main():
    summary = {}
    for ds in DATASETS:
        with open(f"{DATA_ROOT}/{ds}/data-train.json") as f:
            train = json.load(f)
        with open(f"{DATA_ROOT}/{ds}/data-test.json") as f:
            test = json.load(f)
        train_captures = {capture(it["pcap_file"]) for it in train}
        train_connections = {connection(it["pcap_file"]) for it in train} - {None}
        captures_per_class = defaultdict(set)
        for it in train + test:
            captures_per_class[it["label"]].add(capture(it["pcap_file"]))
        test_connections = [connection(it["pcap_file"]) for it in test]
        unseen_capture = {i for i, it in enumerate(test) if capture(it["pcap_file"]) not in train_captures}
        unseen_connection = {i for i, c in enumerate(test_connections) if c is not None and c not in train_connections}
        parsed = sum(c is not None for c in test_connections)
        entry = {
            "test_flows": len(test),
            "test_flows_from_training_capture": len(test) - len(unseen_capture),
            "test_flows_with_parsed_endpoints": parsed,
            "test_flows_whose_connection_is_in_training": parsed - len(unseen_connection),
            "classes": len(captures_per_class),
            "classes_with_single_capture_file": sum(1 for c in captures_per_class.values() if len(c) == 1),
        }
        summary[ds] = entry
        print(f"{ds:15s} test={entry['test_flows']} from-training-capture={entry['test_flows_from_training_capture']} "
              f"connection-in-training={entry['test_flows_whose_connection_is_in_training']}/{parsed} "
              f"single-capture classes={entry['classes_with_single_capture_file']}/{entry['classes']}", flush=True)
    os.makedirs(RESULTS_OUT, exist_ok=True)
    with open(f"{RESULTS_OUT}/capture_overlap.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {RESULTS_OUT}/capture_overlap.json")


if __name__ == "__main__":
    main()
