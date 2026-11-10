"""NetMamba-N: a NetMamba-style classifier trained separately for each packet count N.

The input is the first N packet records of the flows with at least N packets, scaled to
[-1, 1]. A Conv1d patch embedding with stride 4, a learned position embedding, a class
token appended after the patches, unidirectional Mamba blocks and a linear head give the
class logits. Training follows the common recipe of run_flowmamba_early.py. Test scores
are appended to OUTPUT_ROOT/results/netmamba_n_baseline_summary.csv:

    python src/run_netmamba_n_baseline.py --dataset iscxvpn --n_packets 3 --seed 20260921
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader, Dataset

from paths import DATA_ROOT, OUTPUT_ROOT, RESULTS_OUT
from run_flowmamba_early import string_list_to_arr

NUM_PACKET_BYTE = 320


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
        self.n_packets = n_packets
        self.labels = np.array([item["label"] for item in raw], dtype=np.int64)
        self.bytes = np.stack([
            np.array(string_list_to_arr(item["data"], num_string=n_packets), dtype=np.float32) for item in raw
        ])
        self.bytes = (self.bytes / 255.0 - 0.5) / 0.5

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.bytes[idx]).unsqueeze(0)
        y = int(self.labels[idx])
        return x, y


class MambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = Mamba(d_model=d_model, layer_idx=0)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class NetMambaLite(nn.Module):
    def __init__(self, arr_length, stride_size=4, embed_dim=256, depth=4, num_classes=10):
        super().__init__()
        assert arr_length % stride_size == 0
        num_patches = arr_length // stride_size
        self.patch_embed = nn.Conv1d(1, embed_dim, kernel_size=stride_size, stride=stride_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList([MambaBlock(embed_dim) for _ in range(depth)])
        self.norm_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).transpose(1, 2)
        x = x + self.pos_embed[:, :-1, :]
        cls = self.cls_token + self.pos_embed[:, -1:, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([x, cls], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        cls_out = x[:, -1, :]
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

    run_id = f"netmamba_n{args.n_packets}_{args.dataset}_seed{args.seed}"
    run_dir = f"{OUTPUT_ROOT}/runs/{args.dataset}/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    arr_length = args.n_packets * NUM_PACKET_BYTE
    model = NetMambaLite(arr_length=arr_length, num_classes=nb_classes).to(device)
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
            x = x.to(device)
            y = y.to(device)
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
    summary_path = f"{RESULTS_OUT}/netmamba_n_baseline_summary.csv"
    write_header = not os.path.exists(summary_path)
    cols = ["dataset", "n_packets", "seed", "nb_classes", "n_train", "n_valid", "n_test",
            "best_val_f1", "test_acc", "test_macro_f1", "test_mcc", "elapsed_sec", "epochs_ran"]
    with open(summary_path, "a") as f:
        if write_header:
            f.write(",".join(cols) + "\n")
        f.write(",".join(str(result[k]) for k in cols) + "\n")


if __name__ == "__main__":
    main()
