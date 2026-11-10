"""FlowMamba-Early with exit-specific layers: a shared trunk of n_shared Mamba-2 blocks
followed, for every exit, by its own branch of n_branch Mamba-2 blocks.

Everything else follows run_flowmamba_early.py. Test scores are appended to
OUTPUT_ROOT/results/flowmamba_v3_summary.csv:

    python src/run_branched_exits.py --dataset ustctfc2016 --seed 20260921 --n_shared 2 --n_branch 2
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
ARR_LENGTH = NUM_PACKET * NUM_PACKET_BYTE  # 1600
EXIT_POSITIONS = [1, 2, 3, 5]
D_MODEL = 256


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


def make_mamba_layer(d_model, device):
    return Mamba2(d_model, headdim=16, d_state=16, chunk_size=256, device=device)


class FlowMambaEarlyV3(nn.Module):
    """Shared trunk (n_shared Mamba2 layers) + per-exit independent branch
    (n_branch Mamba2 layers each). Each branch only ever sees the causal
    prefix up to its own exit position, so no future-byte leakage is
    possible by construction (branches are separate modules, not shared
    weights re-applied to a longer context)."""

    def __init__(self, vocab=256, d_model=D_MODEL, num_classes=10,
                 exit_positions=EXIT_POSITIONS, bytes_per_pkt=NUM_PACKET_BYTE,
                 n_shared=2, n_branch=2, device="cuda"):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.shared_layers = nn.ModuleList([make_mamba_layer(d_model, device) for _ in range(n_shared)])
        self.shared_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_shared)])

        self.exit_positions = exit_positions
        self.bytes_per_pkt = bytes_per_pkt
        self.branch_layers = nn.ModuleList([
            nn.ModuleList([make_mamba_layer(d_model, device) for _ in range(n_branch)])
            for _ in exit_positions
        ])
        self.branch_norms = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_branch)])
            for _ in exit_positions
        ])
        self.branch_norm_f = nn.ModuleList([nn.LayerNorm(d_model) for _ in exit_positions])
        self.heads = nn.ModuleList([nn.Linear(d_model, num_classes) for _ in exit_positions])

    def forward(self, x):
        h0 = self.emb(x)
        for layer, norm in zip(self.shared_layers, self.shared_norms):
            h0 = h0 + layer(norm(h0))

        outs = []
        for i, pos in enumerate(self.exit_positions):
            idx = pos * self.bytes_per_pkt - 1
            hb = h0[:, :idx + 1]
            for layer, norm in zip(self.branch_layers[i], self.branch_norms[i]):
                hb = hb + layer(norm(hb))
            hb = self.branch_norm_f[i](hb)
            feat = hb[:, -1]
            outs.append(self.heads[i](feat))
        return outs


def masked_train_loss(outs, y, npkt, exit_positions):
    losses = []
    for pos, logits in zip(exit_positions, outs):
        mask = npkt >= pos
        if mask.sum() > 0:
            losses.append(F.cross_entropy(logits[mask], y[mask]))
    if not losses:
        return None
    return sum(losses) / len(losses)


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


def format_val_line(epoch, mean_loss, val_score, val_results, exit_positions):
    parts = []
    for p in exit_positions:
        f1_val = val_results[p]["f1"]
        n_val = val_results[p]["n"]
        parts.append("val_f1@" + str(p) + "=" + format(f1_val, ".4f") + "(n=" + str(n_val) + ")")
    return "epoch " + str(epoch) + " train_loss " + format(mean_loss, ".4f") + " val_mean_f1 " + format(val_score, ".4f") + " " + " ".join(parts)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--n_shared", type=int, default=2)
    ap.add_argument("--n_branch", type=int, default=2)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nb_classes = get_nb_classes(args.dataset)
    train_loader, valid_loader, test_loader = build_loaders(args.dataset, args.seed, args.batch_size)

    run_id = f"flowmamba_v3_{args.dataset}_seed{args.seed}_s{args.n_shared}b{args.n_branch}"
    run_dir = f"{OUTPUT_ROOT}/runs/{args.dataset}/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    model = FlowMambaEarlyV3(num_classes=nb_classes, n_shared=args.n_shared, n_branch=args.n_branch, device=device).to(device)
    n_params = count_params(model)
    print(f"MODEL_PARAMS n_shared={args.n_shared} n_branch={args.n_branch} total_params={n_params}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val_score = -1
    best_state = None
    bad_epochs = 0
    log_lines = [f"MODEL_PARAMS n_shared={args.n_shared} n_branch={args.n_branch} total_params={n_params}"]
    t0 = time.time()
    epoch = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for x, y, npkt in train_loader:
            x, y, npkt = x.to(device), y.to(device), npkt.to(device)
            opt.zero_grad()
            outs = model(x)
            loss = masked_train_loss(outs, y, npkt, EXIT_POSITIONS)
            if loss is None:
                continue
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        val_results = evaluate(model, valid_loader, device, EXIT_POSITIONS)
        val_score = float(np.mean([val_results[p]["f1"] for p in EXIT_POSITIONS]))
        line = format_val_line(epoch, total_loss / max(n, 1), val_score, val_results, EXIT_POSITIONS)
        print(line, flush=True)
        log_lines.append(line)
        if val_score > best_val_score:
            best_val_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                log_lines.append(f"early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, f"{run_dir}/checkpoint-best.pth")
    test_results = evaluate(model, test_loader, device, EXIT_POSITIONS)
    elapsed = time.time() - t0

    result = {
        "dataset": args.dataset, "seed": args.seed, "n_shared": args.n_shared, "n_branch": args.n_branch,
        "n_params": n_params, "nb_classes": nb_classes,
        "n_train": len(train_loader.dataset), "n_valid": len(valid_loader.dataset), "n_test": len(test_loader.dataset),
        "best_val_mean_f1": best_val_score, "elapsed_sec": elapsed, "epochs_ran": epoch + 1,
        "per_exit": {str(p): test_results[p] for p in EXIT_POSITIONS},
    }
    with open(f"{run_dir}/run_log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n" + json.dumps(result, indent=2))
    print("RESULT_JSON " + json.dumps(result), flush=True)

    os.makedirs(RESULTS_OUT, exist_ok=True)
    summary_path = f"{RESULTS_OUT}/flowmamba_v3_summary.csv"
    write_header = not os.path.exists(summary_path)
    cols = ["dataset", "seed", "n_shared", "n_branch", "n_params", "exit_pos", "nb_classes", "n_train", "n_valid", "n_test_eligible",
            "test_acc", "test_macro_f1", "test_mcc", "elapsed_sec", "epochs_ran"]
    with open(summary_path, "a") as f:
        if write_header:
            f.write(",".join(cols) + "\n")
        for p in EXIT_POSITIONS:
            row = {
                "dataset": args.dataset, "seed": args.seed, "n_shared": args.n_shared, "n_branch": args.n_branch,
                "n_params": n_params, "exit_pos": p, "nb_classes": nb_classes,
                "n_train": len(train_loader.dataset), "n_valid": len(valid_loader.dataset),
                "n_test_eligible": test_results[p]["n"],
                "test_acc": test_results[p]["acc"], "test_macro_f1": test_results[p]["f1"],
                "test_mcc": test_results[p]["mcc"], "elapsed_sec": elapsed, "epochs_ran": epoch + 1,
            }
            f.write(",".join(str(row[c]) for c in cols) + "\n")


if __name__ == "__main__":
    main()
