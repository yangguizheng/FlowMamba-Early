"""Training jobs of the controlled comparison: FlowMamba-Early (FME) and FlowMamba-Single-N (Single).

A job is a dict with the keys
    model        "FME" or "Single"
    variant      name of the run family, used in the output path (for example orig, stepmatch,
                 maskall, elig3, long50)
    dataset      iscxvpn, ciciot2022, ustctfc2016, cipherspectrum or cstnet-tls1.3
    seed         random seed
    n            exit of a single-exit model; for FME, the minimum packet count of the training
                 flows (eligibility-matched model), or None for all training flows
    budget       "orig" (25 epochs, patience 6), "long50" (50 epochs, patience 12) or "stepmatch"
                 (min(ceil(25 / rho), 100) epochs and min(ceil(6 / rho), 24) patience, where rho is
                 the share of training flows with at least n packets)
    mask         None or a mask of masks.py applied to training, validation and test inputs
    epoch_preds  save the validation and test predictions of every epoch

The model, byte encoding, masked multi-exit loss, optimizer, schedule and early stopping are
those of run_flowmamba_early.py and run_flowmamba_single_n.py; with budget "orig" and no mask
a job repeats those scripts. The cosine period equals the number of epochs. Checkpoints are
selected on validation scores only: the mean macro-F1 over the exits for FME and the exit's
macro-F1 for Single. Each run writes config.json, the best checkpoint, per-flow predictions of
the selected epoch (preds_best.npz) and, last, result.json, which marks the run as complete.
"""
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

import run_flowmamba_early as orig
from paths import DATA_ROOT, OUTPUT_ROOT, REPO

OUT = f"{OUTPUT_ROOT}/experiments"
EXITS = list(orig.EXIT_POSITIONS)          # [1, 2, 3, 5]
PKT = orig.NUM_PACKET_BYTE                 # 320
DATASETS = ["iscxvpn", "ciciot2022", "ustctfc2016", "cipherspectrum", "cstnet-tls1.3"]
NAMES = {"iscxvpn": "ISCXVPN2016", "ciciot2022": "CICIoT2022", "ustctfc2016": "USTC-TFC2016",
         "cipherspectrum": "CipherSpectrum", "cstnet-tls1.3": "CSTNET-TLS1.3"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split(dataset, split, mask=None):
    """Return arrays for one split: 1,600 input bytes per flow (uint8), labels,
    real packets capped at five, raw packet count, and the source pcap path.
    `mask` is applied to the bytes of every packet record (see masks.py)."""
    path = f"{DATA_ROOT}/{dataset}/data-{split}.json"
    with open(path) as f:
        raw = json.load(f)
    data = np.stack([np.array(orig.string_list_to_arr(it["data"]), dtype=np.uint8) for it in raw])
    out = {
        "bytes": data,
        "labels": np.array([it["label"] for it in raw], dtype=np.int64),
        "npkt": np.array([min(it["num_packet"], orig.NUM_PACKET) for it in raw], dtype=np.int64),
        "npkt_raw": np.array([it["num_packet"] for it in raw], dtype=np.int64),
        "pcap": [it["pcap_file"] for it in raw],
        "index": np.arange(len(raw), dtype=np.int64),
        "sha256": sha256(path),
    }
    if mask:
        import masks
        out["mask_stats"] = masks.apply(out["bytes"], mask)
    return out


class MultiExitSet(Dataset):
    """Same items as run_flowmamba_early.ByteDatasetMultiExit: int64 bytes, label, capped packet count."""

    def __init__(self, split, keep=None):
        keep = np.arange(len(split["labels"])) if keep is None else keep
        self.bytes, self.labels = split["bytes"][keep], split["labels"][keep]
        self.npkt, self.index = split["npkt"][keep], split["index"][keep]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.from_numpy(self.bytes[i].astype(np.int64)), int(self.labels[i]), int(self.npkt[i])


class TruncatedSet(Dataset):
    """Same items as ByteDatasetTruncated of run_flowmamba_single_n.py: flows with at
    least n packets, input cut to the first n packets."""

    def __init__(self, split, n):
        keep = np.flatnonzero(split["npkt_raw"] >= n)
        self.bytes = np.ascontiguousarray(split["bytes"][keep, :n * PKT])
        self.labels, self.index = split["labels"][keep], split["index"][keep]
        self.npkt = split["npkt"][keep]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.from_numpy(self.bytes[i].astype(np.int64)), int(self.labels[i])


def macro_f1(y, p):
    return float(f1_score(y, p, average="macro", zero_division=0))


@torch.no_grad()
def predict(model, loader, exits, multi):
    """Per-exit predictions on the eligible flows: dataset positions, labels,
    predicted class, and top-1 probability."""
    model.eval()
    acc = {e: {"pos": [], "y": [], "p": [], "prob": []} for e in exits}
    offset = 0
    for batch in loader:
        x, y = batch[0].to("cuda"), batch[1].numpy()
        npkt = batch[2].numpy() if multi else None
        outs = model(x)
        for e, logits in zip(exits, outs):
            pred = logits.argmax(-1)  # as in run_flowmamba_early.evaluate()
            prob = logits.float().softmax(-1).gather(1, pred[:, None]).squeeze(1)
            keep = np.flatnonzero(npkt >= e) if multi else np.arange(len(y))
            acc[e]["pos"].append(offset + keep)
            acc[e]["y"].append(y[keep])
            acc[e]["p"].append(pred.cpu().numpy()[keep])
            acc[e]["prob"].append(prob.cpu().numpy()[keep])
        offset += len(y)
    return {e: {k: np.concatenate(v[k]) for k in v} for e, v in acc.items()}


def scores(pred):
    return {e: {"n": int(len(v["y"])), "f1": macro_f1(v["y"], v["p"]),
                "acc": float(accuracy_score(v["y"], v["p"]))} for e, v in pred.items()}


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def env_record():
    """Software, hardware and source versions stored with every run."""
    return {
        "git_commit": git("rev-parse", "HEAD"), "src_worktree_clean": git("status", "--porcelain", "src") == "",
        "source_sha256": {n: sha256(f"{REPO}/src/{n}") for n in ["run_flowmamba_early.py", "run_flowmamba_single_n.py", "experiment.py"]},
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "python": platform.python_version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def run_dir_for(job):
    tag = f"{job['model']}_{job['variant']}_{job['dataset']}" + (f"_n{job['n']}" if job.get("n") else "") + f"_s{job['seed']}"
    return f"{OUT}/runs/{job['variant']}/{job['dataset']}/{tag}", tag


def rho(train, n):
    return float((train["npkt_raw"] >= n).mean())


def budget(job, train):
    """Epochs, patience and cosine period for the job's training budget."""
    if job["budget"] == "orig":
        return 25, 6
    if job["budget"] == "long50":
        return 50, 12
    if job["budget"] == "stepmatch":
        r = rho(train, job["n"])
        return min(math.ceil(25 / r), 100), min(math.ceil(6 / r), 24)
    raise ValueError(job["budget"])


def train(job):
    """Train one job (see the module docstring) and write its run directory."""
    run_dir, tag = run_dir_for(job)
    os.makedirs(run_dir, exist_ok=True)
    start = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    orig.seed_everything(job["seed"])
    nb = orig.get_nb_classes(job["dataset"])
    split = {s: load_split(job["dataset"], s, job.get("mask")) for s in ["train", "valid", "test"]}
    epochs, patience = budget(job, split["train"])
    multi = job["model"] == "FME"
    if multi:
        keep = None if not job.get("n") else np.flatnonzero(split["train"]["npkt_raw"] >= job["n"])
        sets = {"train": MultiExitSet(split["train"], keep), "valid": MultiExitSet(split["valid"]), "test": MultiExitSet(split["test"])}
        exits = EXITS
    else:
        sets = {s: TruncatedSet(split[s], job["n"]) for s in split}
        exits = [job["n"]]
    g = torch.Generator()
    g.manual_seed(job["seed"])
    loaders = {"train": DataLoader(sets["train"], batch_size=128, shuffle=True, generator=g, num_workers=2, drop_last=False),
               "valid": DataLoader(sets["valid"], batch_size=128, shuffle=False, num_workers=2),
               "test": DataLoader(sets["test"], batch_size=128, shuffle=False, num_workers=2)}

    model = orig.FlowMambaEarly(num_classes=nb, exit_positions=exits, device="cuda").to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    config = dict(job, run_id=tag, epochs=epochs, patience=patience, t_max=epochs, batch_size=128, lr=1e-3,
                  weight_decay=0.05, optimizer="AdamW", scheduler="CosineAnnealingLR", precision="fp32",
                  exits=exits, nb_classes=nb, checkpoint_selection="mean_validation_macro_f1" if multi else "exit_validation_macro_f1",
                  train_flows=len(sets["train"]), valid_flows=len(sets["valid"]), test_flows=len(sets["test"]),
                  rho=rho(split["train"], job["n"]) if job.get("n") else 1.0,
                  data_sha256={s: split[s]["sha256"] for s in split}, start_utc=start, status="running", **env_record())
    if job.get("mask"):
        config["mask_stats"] = {s: split[s]["mask_stats"] for s in split}
    json.dump(config, open(f"{run_dir}/config.json", "w"), indent=2)

    best, best_state, best_epoch, bad, steps, lines, history = -1, None, -1, 0, 0, [], []
    epoch_preds = {"valid": [], "test": []}
    epoch = 0
    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for batch in loaders["train"]:
            x, y = batch[0].to("cuda"), batch[1].to("cuda")
            opt.zero_grad()
            outs = model(x)
            if multi:
                loss = orig.masked_train_loss(outs, y, batch[2].to("cuda"), exits)
                if loss is None:
                    continue
            else:
                loss = criterion(outs[0], y)
            loss.backward()
            opt.step()
            steps += 1
            total += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        vp = predict(model, loaders["valid"], exits, multi)
        vs = scores(vp)
        score = float(np.mean([vs[e]["f1"] for e in exits]))
        if multi:
            line = orig.format_val_line(epoch, total / max(n, 1), score, {e: {"f1": vs[e]["f1"], "n": vs[e]["n"]} for e in exits}, exits)
        else:
            line = f"epoch {epoch} train_loss {total / max(n, 1):.4f} val_acc {vs[exits[0]]['acc']:.4f} val_f1 {score:.4f}"
        print(line, flush=True)
        lines.append(line)
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_score": score,
                        "val": {str(e): vs[e] for e in exits}})
        if job.get("epoch_preds"):
            tp = predict(model, loaders["test"], exits, multi)
            epoch_preds["valid"].append(vp)
            epoch_preds["test"].append(tp)
        if score > best:
            best, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                lines.append(f"early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, f"{run_dir}/checkpoint-best.pth")
    final = {s: predict(model, loaders[s], exits, multi) for s in ["valid", "test"]}
    arrays = {}
    for s, pred in final.items():
        index = sets[s].index
        for e, v in pred.items():
            arrays[f"{s}_e{e}_index"] = index[v["pos"]]
            arrays[f"{s}_e{e}_y"] = v["y"].astype(np.int16)
            arrays[f"{s}_e{e}_pred"] = v["p"].astype(np.int16)
            arrays[f"{s}_e{e}_prob"] = v["prob"].astype(np.float32)
            arrays[f"{s}_e{e}_npkt"] = sets[s].npkt[v["pos"]].astype(np.int8)
    np.savez_compressed(f"{run_dir}/preds_best.npz", **arrays)
    if job.get("epoch_preds"):
        ep = {}
        for s in ["valid", "test"]:
            for e in exits:
                ep[f"{s}_e{e}_index"] = sets[s].index[epoch_preds[s][0][e]["pos"]]
                ep[f"{s}_e{e}_y"] = epoch_preds[s][0][e]["y"].astype(np.int16)
                ep[f"{s}_e{e}_pred"] = np.stack([p[e]["p"] for p in epoch_preds[s]]).astype(np.int16)
                ep[f"{s}_e{e}_prob"] = np.stack([p[e]["prob"] for p in epoch_preds[s]]).astype(np.float16)
        np.savez_compressed(f"{run_dir}/preds_epochs.npz", **ep)
    test = scores(final["test"])
    result = {"run_id": tag, "best_val_score": best, "best_epoch": best_epoch, "epochs_run": epoch + 1,
              "optimizer_steps": steps, "elapsed_sec": time.time() - t0,
              "test": {str(e): test[e] for e in exits}, "history": history}
    with open(f"{run_dir}/run_log.txt", "w") as f:
        f.write("\n".join(lines) + "\n" + json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    config.update(status="ok", end_utc=datetime.now(timezone.utc).isoformat(), epochs_run=epoch + 1,
                  best_epoch=best_epoch, optimizer_steps=steps, wall_time_min=(time.time() - t0) / 60)
    json.dump(config, open(f"{run_dir}/config.json", "w"), indent=2)
    json.dump(result, open(f"{run_dir}/result.json", "w"), indent=2)   # written last: completion marker
    print("RESULT_JSON " + json.dumps({k: v for k, v in result.items() if k != "history"}), flush=True)
    return result
