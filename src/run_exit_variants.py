"""Variants of the exit layout and loss weighting of FlowMamba-Early.

--variant adapter      one Linear+GELU projection per exit instead of the shared projection
--variant uncertainty  exit losses weighted by learned log-variances (homoscedastic uncertainty weighting)
--variant both         both changes
--variant baseline     the shared projection and the uniform mean of the exit losses

Everything else follows run_flowmamba_early.py. Test scores are appended to
OUTPUT_ROOT/results/flowmamba_v2_summary.csv:

    python src/run_exit_variants.py --dataset ustctfc2016 --seed 20260921 --variant adapter
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
ARR_LENGTH = NUM_PACKET * NUM_PACKET_BYTE
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


class FlowMambaEarlyV2(nn.Module):
    def __init__(self, vocab=256, d_model=D_MODEL, n_layers=N_LAYERS, num_classes=10,
                 exit_positions=EXIT_POSITIONS, bytes_per_pkt=NUM_PACKET_BYTE, device="cuda",
                 use_adapter=False):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([
            Mamba2(d_model, headdim=16, d_state=16, chunk_size=256, device=device) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.exit_positions = exit_positions
        self.bytes_per_pkt = bytes_per_pkt
        self.use_adapter = use_adapter
        if use_adapter:
            self.adapters = nn.ModuleList([
                nn.Sequential(nn.Linear(d_model, d_model), nn.GELU()) for _ in exit_positions
            ])
        else:
            self.shared_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.heads = nn.ModuleList([nn.Linear(d_model, num_classes) for _ in exit_positions])

    def forward(self, x):
        h = self.emb(x)
        for layer, norm in zip(self.layers, self.norms):
            h = h + layer(norm(h))
        h = self.norm_f(h)
        outs = []
        for i, (pos, head) in enumerate(zip(self.exit_positions, self.heads)):
            idx = pos * self.bytes_per_pkt - 1
            hs = h[:, idx]
            feat = self.adapters[i](hs) if self.use_adapter else self.shared_mlp(hs)
            outs.append(head(feat))
        return outs


class UncertaintyWeights(nn.Module):
    def __init__(self, n_exits):
        super().__init__()
        self.log_var = nn.Parameter(torch.zeros(n_exits))

    def forward(self, per_exit_losses):
        # per_exit_losses: list aligned with self.log_var order (may skip some if empty batch)
        total = 0.0
        for i, l in per_exit_losses:
            total = total + torch.exp(-self.log_var[i]) * l + self.log_var[i]
        return total


def masked_losses(outs, y, npkt, exit_positions):
    """Returns list of (idx_in_exit_positions, loss) for exits with eligible samples."""
    result = []
    for i, (pos, logits) in enumerate(zip(exit_positions, outs)):
        mask = npkt >= pos
        if mask.sum() > 0:
            result.append((i, F.cross_entropy(logits[mask], y[mask])))
    return result


def combine_uniform(per_exit_losses):
    if not per_exit_losses:
        return None
    return sum(l for _, l in per_exit_losses) / len(per_exit_losses)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--variant", choices=["baseline", "adapter", "uncertainty", "both"], default="baseline")
    args = ap.parse_args()

    use_adapter = args.variant in ("adapter", "both")
    use_uncertainty = args.variant in ("uncertainty", "both")

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nb_classes = get_nb_classes(args.dataset)
    train_loader, valid_loader, test_loader = build_loaders(args.dataset, args.seed, args.batch_size)

    run_id = f"flowmamba_v2_{args.variant}_{args.dataset}_seed{args.seed}"
    run_dir = f"{OUTPUT_ROOT}/runs/{args.dataset}/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    model = FlowMambaEarlyV2(num_classes=nb_classes, device=device, use_adapter=use_adapter).to(device)
    params = list(model.parameters())
    uw = None
    if use_uncertainty:
        uw = UncertaintyWeights(len(EXIT_POSITIONS)).to(device)
        params += list(uw.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val_score = -1
    best_state = None
    bad_epochs = 0
    log_lines = []
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
            per_exit = masked_losses(outs, y, npkt, EXIT_POSITIONS)
            if not per_exit:
                continue
            if use_uncertainty:
                loss = uw(per_exit)
            else:
                loss = combine_uniform(per_exit)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        val_results = evaluate(model, valid_loader, device, EXIT_POSITIONS)
        val_score = float(np.mean([val_results[p]["f1"] for p in EXIT_POSITIONS]))
        line = format_val_line(epoch, total_loss / max(n, 1), val_score, val_results, EXIT_POSITIONS)
        if use_uncertainty:
            line += " log_var=" + str([round(float(v), 3) for v in uw.log_var.detach().cpu().tolist()])
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
        "dataset": args.dataset, "seed": args.seed, "variant": args.variant, "nb_classes": nb_classes,
        "n_train": len(train_loader.dataset), "n_valid": len(valid_loader.dataset), "n_test": len(test_loader.dataset),
        "best_val_mean_f1": best_val_score, "elapsed_sec": elapsed, "epochs_ran": epoch + 1,
        "per_exit": {str(p): test_results[p] for p in EXIT_POSITIONS},
    }
    with open(f"{run_dir}/run_log.txt", "w") as f:
        f.write("\n".join(log_lines) + "\n" + json.dumps(result, indent=2))
    print("RESULT_JSON " + json.dumps(result), flush=True)

    os.makedirs(RESULTS_OUT, exist_ok=True)
    summary_path = f"{RESULTS_OUT}/flowmamba_v2_summary.csv"
    write_header = not os.path.exists(summary_path)
    cols = ["dataset", "seed", "variant", "exit_pos", "nb_classes", "n_train", "n_valid", "n_test_eligible",
            "test_acc", "test_macro_f1", "test_mcc", "elapsed_sec", "epochs_ran"]
    with open(summary_path, "a") as f:
        if write_header:
            f.write(",".join(cols) + "\n")
        for p in EXIT_POSITIONS:
            row = {
                "dataset": args.dataset, "seed": args.seed, "variant": args.variant, "exit_pos": p, "nb_classes": nb_classes,
                "n_train": len(train_loader.dataset), "n_valid": len(valid_loader.dataset),
                "n_test_eligible": test_results[p]["n"],
                "test_acc": test_results[p]["acc"], "test_macro_f1": test_results[p]["f1"],
                "test_mcc": test_results[p]["mcc"], "elapsed_sec": elapsed, "epochs_ran": epoch + 1,
            }
            f.write(",".join(str(row[c]) for c in cols) + "\n")


if __name__ == "__main__":
    main()
