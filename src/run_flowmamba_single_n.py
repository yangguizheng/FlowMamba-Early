"""FlowMamba-Single-N: a single-exit model of the same architecture, trained for one packet count N.

The model is FlowMambaEarly from run_flowmamba_early.py with one exit at packet N. It is
trained on the flows with at least N packets, with the input cut to the first N packets.
Because the backbone is causal, reading byte N*320-1 of the cut input gives the same value
as reading it from the full five-packet input. The recipe is that of run_flowmamba_early.py;
the differences are the single-exit loss, the training flows, and checkpoint selection by
this exit's validation macro-F1. Test scores are appended to
OUTPUT_ROOT/results/flowmamba_single_n_summary.csv:

    python src/run_flowmamba_single_n.py --dataset iscxvpn --n_packets 3 --seed 20260921
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader, Dataset

from paths import DATA_ROOT, OUTPUT_ROOT
from run_flowmamba_early import FlowMambaEarly, get_nb_classes, seed_everything, string_list_to_arr


class ByteDatasetTruncated(Dataset):
    """Flows with at least n_packets packets, cut to their first n_packets packet records."""

    def __init__(self, path, n_packets):
        with open(path) as f:
            raw = json.load(f)
        # position of each kept flow in the source json, so saved predictions can be re-subset later
        self.index = np.array([i for i, item in enumerate(raw) if item["num_packet"] >= n_packets], dtype=np.int64)
        raw = [raw[i] for i in self.index]
        self.labels = np.array([item["label"] for item in raw], dtype=np.int64)
        self.bytes = np.stack([
            np.array(string_list_to_arr(item["data"], num_string=n_packets), dtype=np.int64) for item in raw
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.bytes[idx])
        y = int(self.labels[idx])
        return x, y


def build_loaders(dataset, seed, batch_size, n_packets):
    ds_train = ByteDatasetTruncated(f"{DATA_ROOT}/{dataset}/data-train.json", n_packets)
    ds_valid = ByteDatasetTruncated(f"{DATA_ROOT}/{dataset}/data-valid.json", n_packets)
    ds_test = ByteDatasetTruncated(f"{DATA_ROOT}/{dataset}/data-test.json", n_packets)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, generator=g, num_workers=2, drop_last=False)
    valid_loader = DataLoader(ds_valid, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, valid_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)[0]
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1m = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    try:
        mcc = matthews_corrcoef(all_labels, all_preds)
    except ValueError:
        mcc = float("nan")
    return acc, f1m, mcc, all_preds, all_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n_packets", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--out_root", default=OUTPUT_ROOT, help="directory that receives runs/ and results/")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nb_classes = get_nb_classes(args.dataset)
    train_loader, valid_loader, test_loader = build_loaders(args.dataset, args.seed, args.batch_size, args.n_packets)

    run_id = f"flowmamba_single_n{args.n_packets}_{args.dataset}_seed{args.seed}"
    run_dir = f"{args.out_root}/runs/{args.dataset}/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    model = FlowMambaEarly(num_classes=nb_classes, exit_positions=[args.n_packets], device=device).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1
    best_state = None
    bad_epochs = 0
    n_updates = 0
    log_lines = []
    t0 = time.time()
    epoch = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)[0]
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            n_updates += 1
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        val_acc, val_f1, val_mcc, _, _ = evaluate(model, valid_loader, device)
        line = f"epoch {epoch} train_loss {total_loss / max(n,1):.4f} val_acc {val_acc:.4f} val_f1 {val_f1:.4f} val_mcc {val_mcc:.4f}"
        print(line, flush=True)
        log_lines.append(line)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                log_lines.append(f"early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, f"{run_dir}/checkpoint-best.pth")
    test_acc, test_f1, test_mcc, test_preds, test_labels = evaluate(model, test_loader, device)
    np.savez(f"{run_dir}/test_predictions.npz", index=test_loader.dataset.index, y_true=test_labels, y_pred=test_preds)
    elapsed = time.time() - t0

    result = {
        "dataset": args.dataset, "n_packets": args.n_packets, "seed": args.seed, "nb_classes": nb_classes,
        "n_train": len(train_loader.dataset), "n_valid": len(valid_loader.dataset), "n_test": len(test_loader.dataset),
        "best_val_f1": best_val_f1,
        "test_acc": test_acc, "test_macro_f1": test_f1, "test_mcc": test_mcc,
        "elapsed_sec": elapsed, "epochs_ran": epoch + 1, "n_updates": n_updates,
    }
    with open(f"{run_dir}/run_log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n" + json.dumps(result, indent=2))
    print("RESULT_JSON " + json.dumps(result), flush=True)

    summary_path = f"{args.out_root}/results/flowmamba_single_n_summary.csv"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    write_header = not os.path.exists(summary_path)
    cols = ["dataset", "n_packets", "seed", "nb_classes", "n_train", "n_valid", "n_test",
            "best_val_f1", "test_acc", "test_macro_f1", "test_mcc", "elapsed_sec", "epochs_ran", "n_updates"]
    with open(summary_path, "a") as f:
        if write_header:
            f.write(",".join(cols) + "\n")
        f.write(",".join(str(result[k]) for k in cols) + "\n")


if __name__ == "__main__":
    main()
