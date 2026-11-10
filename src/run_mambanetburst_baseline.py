"""MNB: a MambaNetBurst-style byte classifier trained separately for each packet count N.

The input is the first N packet records of the flows with at least N packets. Every byte is
a token, Mamba-2 blocks process the sequence, and the class is read from a token appended
after the bytes. Training follows the common recipe of run_flowmamba_early.py. Test scores
are appended to OUTPUT_ROOT/results/mambanetburst_summary.csv:

    python src/run_mambanetburst_baseline.py --dataset iscxvpn --n_packets 3 --seed 20260921
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba2 import Mamba2
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader, Dataset

from paths import DATA_ROOT, OUTPUT_ROOT, RESULTS_OUT
from run_flowmamba_early import string_list_to_arr

NUM_PACKET_BYTE = 320
D_MODEL = 256
N_LAYERS = 4


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ByteDatasetTruncated(Dataset):
    def __init__(self, path, n_packets):
        with open(path) as f:
            raw = json.load(f)
        raw = [item for item in raw if item["num_packet"] >= n_packets]
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


class MambaNetBurst(nn.Module):
    """Byte-level, tokenization-free Mamba-2 classifier with a trailing CLS token.

    No patching/striding (unlike NetMamba-N's Conv1d front end): every raw byte
    is its own token, following the design choice in arXiv:2605.11034. The CLS
    token is appended (not prepended) since Mamba's SSM recurrence is causal and
    a leading CLS token could not attend to any subsequent byte.
    """

    def __init__(self, vocab=256, d_model=D_MODEL, n_layers=N_LAYERS, num_classes=10, device="cuda"):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.layers = nn.ModuleList([
            Mamba2(d_model, headdim=16, d_state=16, chunk_size=256, device=device) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        h = self.emb(x)
        cls = self.cls_token.expand(h.shape[0], -1, -1)
        h = torch.cat([h, cls], dim=1)
        for layer, norm in zip(self.layers, self.norms):
            h = h + layer(norm(h))
        h = self.norm_f(h)
        cls_out = h[:, -1, :]
        return self.head(cls_out)


def get_nb_classes(dataset):
    with open(f"{DATA_ROOT}/{dataset}/metadata.json") as f:
        meta = json.load(f)
    return max(meta["name_to_idx"].values()) + 1


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
            logits = model(x)
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
    return acc, f1m, mcc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n_packets", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nb_classes = get_nb_classes(args.dataset)
    train_loader, valid_loader, test_loader = build_loaders(args.dataset, args.seed, args.batch_size, args.n_packets)

    run_id = f"mambanetburst_n{args.n_packets}_{args.dataset}_seed{args.seed}"
    run_dir = f"{OUTPUT_ROOT}/runs/{args.dataset}/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    model = MambaNetBurst(num_classes=nb_classes, device=device).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1
    best_state = None
    bad_epochs = 0
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
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        val_acc, val_f1, val_mcc = evaluate(model, valid_loader, device)
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
    test_acc, test_f1, test_mcc = evaluate(model, test_loader, device)
    elapsed = time.time() - t0

    result = {
        "dataset": args.dataset, "n_packets": args.n_packets, "seed": args.seed, "nb_classes": nb_classes,
        "n_train": len(train_loader.dataset), "n_valid": len(valid_loader.dataset), "n_test": len(test_loader.dataset),
        "best_val_f1": best_val_f1,
        "test_acc": test_acc, "test_macro_f1": test_f1, "test_mcc": test_mcc,
        "elapsed_sec": elapsed, "epochs_ran": epoch + 1,
    }
    with open(f"{run_dir}/run_log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n" + json.dumps(result, indent=2))
    print("RESULT_JSON " + json.dumps(result), flush=True)

    os.makedirs(RESULTS_OUT, exist_ok=True)
    summary_path = f"{RESULTS_OUT}/mambanetburst_summary.csv"
    write_header = not os.path.exists(summary_path)
    cols = ["dataset", "n_packets", "seed", "nb_classes", "n_train", "n_valid", "n_test",
            "best_val_f1", "test_acc", "test_macro_f1", "test_mcc", "elapsed_sec", "epochs_ran"]
    with open(summary_path, "a") as f:
        if write_header:
            f.write(",".join(cols) + "\n")
        f.write(",".join(str(result[k]) for k in cols) + "\n")


if __name__ == "__main__":
    main()
