"""Train FlowMamba-Early for a long fixed budget and record the trajectory of every exit.

The model, loss and optimizer follow run_flowmamba_early.py, with a cosine schedule over
all epochs and no early stopping. After every epoch the script records the training
cross-entropy of each exit; validation and test macro-F1 are computed every fifth epoch
and at the last epoch, and repeated in between. The trajectory is written to
OUTPUT_ROOT/results/long_training/<dataset>_seed<seed>.csv:

    python src/run_long_training.py --dataset ustctfc2016 --seed 20260921 --epochs 150
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba2 import Mamba2
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader, Dataset

from paths import DATA_ROOT, OUTPUT_ROOT, RESULTS_OUT
from run_flowmamba_early import string_list_to_arr

NUM_PACKET = 5
NUM_PACKET_BYTE = 320
EXIT_POSITIONS = [1, 2, 3, 5]
D_MODEL = 256
N_LAYERS = 4


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ByteDatasetMultiExit(Dataset):
    def __init__(self, path):
        with open(path) as f:
            raw = json.load(f)
        self.labels = np.array([item["label"] for item in raw], dtype=np.int64)
        self.num_packet = np.array([min(item["num_packet"], NUM_PACKET) for item in raw], dtype=np.int64)
        self.bytes = np.stack([
            np.array(string_list_to_arr(item["data"]), dtype=np.int64) for item in raw
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.bytes[idx])
        y = int(self.labels[idx])
        npkt = int(self.num_packet[idx])
        return x, y, npkt


class FlowMambaEarly(nn.Module):
    def __init__(self, vocab=256, d_model=D_MODEL, n_layers=N_LAYERS, num_classes=10,
                 exit_positions=EXIT_POSITIONS, bytes_per_pkt=NUM_PACKET_BYTE, device="cuda"):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([
            Mamba2(d_model, headdim=16, d_state=16, chunk_size=256, device=device) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.exit_positions = exit_positions
        self.bytes_per_pkt = bytes_per_pkt
        self.shared_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.heads = nn.ModuleList([nn.Linear(d_model, num_classes) for _ in exit_positions])

    def forward(self, x):
        h = self.emb(x)
        for layer, norm in zip(self.layers, self.norms):
            h = h + layer(norm(h))
        h = self.norm_f(h)
        outs = []
        for pos, head in zip(self.exit_positions, self.heads):
            idx = pos * self.bytes_per_pkt - 1
            feat = self.shared_mlp(h[:, idx])
            outs.append(head(feat))
        return outs


def per_exit_train_ce(outs, y, npkt, exit_positions):
    """Return dict pos -> (sum_ce, count) for this batch, plus the mean loss used for backprop."""
    losses = []
    per_exit = {}
    for pos, logits in zip(exit_positions, outs):
        mask = npkt >= pos
        if mask.sum() > 0:
            ce = F.cross_entropy(logits[mask], y[mask], reduction="sum")
            per_exit[pos] = (ce.item(), int(mask.sum().item()))
            losses.append(F.cross_entropy(logits[mask], y[mask]))
    if not losses:
        return None, per_exit
    return sum(losses) / len(losses), per_exit


def evaluate(model, loader, device, exit_positions):
    model.eval()
    preds_per_exit = {p: [] for p in exit_positions}
    labels_per_exit = {p: [] for p in exit_positions}
    with torch.no_grad():
        for x, y, npkt in loader:
            x = x.to(device)
            outs = model(x)
            npkt_np = npkt.numpy()
            y_np = y.numpy()
            for pos, logits in zip(exit_positions, outs):
                mask = npkt_np >= pos
                if mask.sum() > 0:
                    p_arr = logits.argmax(-1).cpu().numpy()[mask]
                    preds_per_exit[pos].append(p_arr)
                    labels_per_exit[pos].append(y_np[mask])
    results = {}
    for pos in exit_positions:
        if not preds_per_exit[pos]:
            results[pos] = dict(n=0, acc=float("nan"), f1=float("nan"), mcc=float("nan"))
            continue
        preds = np.concatenate(preds_per_exit[pos])
        labels = np.concatenate(labels_per_exit[pos])
        acc = accuracy_score(labels, preds)
        f1m = f1_score(labels, preds, average="macro", zero_division=0)
        try:
            mcc = matthews_corrcoef(labels, preds)
        except ValueError:
            mcc = float("nan")
        results[pos] = dict(n=int(len(labels)), acc=float(acc), f1=float(f1m), mcc=float(mcc))
    return results


def get_nb_classes(dataset):
    with open(f"{DATA_ROOT}/{dataset}/metadata.json") as f:
        meta = json.load(f)
    return max(meta["name_to_idx"].values()) + 1


def build_loaders(dataset, seed, batch_size):
    ds_train = ByteDatasetMultiExit(f"{DATA_ROOT}/{dataset}/data-train.json")
    ds_valid = ByteDatasetMultiExit(f"{DATA_ROOT}/{dataset}/data-valid.json")
    ds_test = ByteDatasetMultiExit(f"{DATA_ROOT}/{dataset}/data-test.json")
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, generator=g, num_workers=2, drop_last=False)
    valid_loader = DataLoader(ds_valid, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, valid_loader, test_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nb_classes = get_nb_classes(args.dataset)
    train_loader, valid_loader, test_loader = build_loaders(args.dataset, args.seed, args.batch_size)

    run_dir = f"{OUTPUT_ROOT}/runs/long_training/{args.dataset}_seed{args.seed}"
    os.makedirs(run_dir, exist_ok=True)

    model = FlowMambaEarly(num_classes=nb_classes, device=device).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    os.makedirs(f"{RESULTS_OUT}/long_training", exist_ok=True)
    csv_path = f"{RESULTS_OUT}/long_training/{args.dataset}_seed{args.seed}.csv"
    cols = ["epoch"] + [f"train_ce_{p}" for p in EXIT_POSITIONS] + [f"val_f1_{p}" for p in EXIT_POSITIONS] + [f"test_f1_{p}" for p in EXIT_POSITIONS]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        sum_ce = {p: 0.0 for p in EXIT_POSITIONS}
        cnt = {p: 0 for p in EXIT_POSITIONS}
        for x, y, npkt in train_loader:
            x, y, npkt = x.to(device), y.to(device), npkt.to(device)
            opt.zero_grad()
            outs = model(x)
            loss, per_exit = per_exit_train_ce(outs, y, npkt, EXIT_POSITIONS)
            if loss is None:
                continue
            loss.backward()
            opt.step()
            for p, (ce_sum, n) in per_exit.items():
                sum_ce[p] += ce_sum
                cnt[p] += n
        sched.step()
        train_ce = {p: (sum_ce[p] / cnt[p] if cnt[p] > 0 else float("nan")) for p in EXIT_POSITIONS}

        # evaluate every fifth epoch and the last one; other rows repeat the latest scores
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            val_results = evaluate(model, valid_loader, device, EXIT_POSITIONS)
            test_results = evaluate(model, test_loader, device, EXIT_POSITIONS)
        row = [str(epoch)]
        row += [format(train_ce[p], ".6f") for p in EXIT_POSITIONS]
        row += [format(val_results[p]["f1"], ".4f") for p in EXIT_POSITIONS]
        row += [format(test_results[p]["f1"], ".4f") for p in EXIT_POSITIONS]
        with open(csv_path, "a") as f:
            f.write(",".join(row) + "\n")
        line = f"epoch {epoch} " + " ".join(f"train_ce_{p}={train_ce[p]:.4f}" for p in EXIT_POSITIONS) + \
               " " + " ".join(f"val_f1_{p}={val_results[p]['f1']:.4f}" for p in EXIT_POSITIONS)
        print(line, flush=True)

    elapsed = time.time() - t0
    with open(f"{run_dir}/DONE", "w") as f:
        f.write(f"elapsed_sec={elapsed}\nepochs={args.epochs}\n")
    print(f"DONE dataset={args.dataset} elapsed_sec={elapsed:.1f}", flush=True)


if __name__ == "__main__":
    main()
