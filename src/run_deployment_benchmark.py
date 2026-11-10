"""Inference cost and incremental-inference agreement of FlowMamba-Early on CSTNET-TLS1.3.

Both parts use the trained 120-class checkpoints and real test flows.

--part accuracy  predictions of the full forward pass and of incremental inference on the
                 whole test split, in fp32 and bf16, for seeds 20260921-23. Writes
                 OUTPUT_ROOT/results/incremental_agreement.json.
--part timing    run on an otherwise idle GPU, in bf16: the time of each exit when the
                 prefix is recomputed, the time of one incremental packet update, and the
                 cumulative time of serving all four exits with FlowMamba-Early and with
                 the four FlowMamba-Single-N checkpoints, each by recomputation or
                 incrementally with one state per model. Also records parameter counts
                 and the per-flow state size. Writes OUTPUT_ROOT/results/deployment_benchmark.json.

The checkpoints come from run_flowmamba_early.py and run_flowmamba_single_n.py; the
parameter counts also read the checkpoints of run_netmamba_n_baseline.py and
run_mambanetburst_baseline.py.

    python src/run_deployment_benchmark.py --part accuracy
    python src/run_deployment_benchmark.py --part timing
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from incremental import IncrementalState, model_forward_incremental_chunk
from paths import DATA_ROOT, OUTPUT_ROOT, RESULTS_OUT
from run_flowmamba_early import (EXIT_POSITIONS, NUM_PACKET, NUM_PACKET_BYTE, ByteDatasetMultiExit, FlowMambaEarly,
                                 get_nb_classes)

DATASET = "cstnet-tls1.3"
SEEDS = [20260921, 20260922, 20260923]
DEVICE = "cuda"
RUNS = f"{OUTPUT_ROOT}/runs/{DATASET}"


def load_fme(seed, nb_classes, dtype):
    model = FlowMambaEarly(num_classes=nb_classes, device=DEVICE)
    model.load_state_dict(torch.load(f"{RUNS}/flowmamba_early_{DATASET}_seed{seed}/checkpoint-best.pth", map_location="cpu"))
    return model.to(DEVICE).to(dtype).eval()


def load_single_n(n, seed, nb_classes, dtype):
    model = FlowMambaEarly(num_classes=nb_classes, exit_positions=[n], device=DEVICE)
    model.load_state_dict(torch.load(f"{RUNS}/flowmamba_single_n{n}_{DATASET}_seed{seed}/checkpoint-best.pth", map_location="cpu"))
    return model.to(DEVICE).to(dtype).eval()


def backbone(model, x):
    h = model.emb(x)
    for layer, norm in zip(model.layers, model.norms):
        h = h + layer(norm(h))
    return model.norm_f(h)


def readout(model, j, h_last):
    return model.heads[j](model.shared_mlp(h_last))


def chunk(x, k):
    return x[:, k * NUM_PACKET_BYTE:(k + 1) * NUM_PACKET_BYTE]


def full_exits(model, x):
    # one pass over all 5 packets; every exit is read from the same pass
    return model(x)


def naive_exits(model, x):
    # recompute: at the decision point of exit N only N packets exist, so the whole prefix is rerun
    return [readout(model, j, backbone(model, x[:, :n * NUM_PACKET_BYTE])[:, -1]) for j, n in enumerate(EXIT_POSITIONS)]


def incremental_exits(model, x):
    # one state per flow, updated once per arriving packet; exits read after packets 1, 2, 3, 5
    state = IncrementalState(len(model.layers))
    outs = []
    for k in range(NUM_PACKET):
        h_last = model_forward_incremental_chunk(model, chunk(x, k), state)
        if k + 1 in EXIT_POSITIONS:
            outs.append(readout(model, EXIT_POSITIONS.index(k + 1), h_last))
    return outs


def specialists_incremental_exits(models, x):
    # the four FlowMamba-Single-N models, each keeping its own state: 1+2+3+5 = 11 packet updates
    outs = []
    for n, model in zip(EXIT_POSITIONS, models):
        state = IncrementalState(len(model.layers))
        for k in range(n):
            h_last = model_forward_incremental_chunk(model, chunk(x, k), state)
        outs.append(readout(model, 0, h_last))
    return outs


def specialists_recompute_exits(models, x):
    # the four FlowMamba-Single-N models, each run once over its own prefix
    return [model(x[:, :n * NUM_PACKET_BYTE])[0] for n, model in zip(EXIT_POSITIONS, models)]


def part_accuracy(out_path):
    nb_classes = get_nb_classes(DATASET)
    ds = ByteDatasetMultiExit(f"{DATA_ROOT}/{DATASET}/data-test.json")
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)
    results = {}
    for seed in SEEDS:
        preds = {}
        for dtype_name, dtype in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
            model = load_fme(seed, nb_classes, dtype)
            for path_name, fn in [("full", full_exits), ("incremental", incremental_exits)]:
                p = []
                with torch.no_grad():
                    for x, _, _ in loader:
                        outs = fn(model, x.to(DEVICE))
                        p.append(torch.stack([o.argmax(-1) for o in outs], dim=1).cpu().numpy())
                preds[f"{dtype_name}_{path_name}"] = np.concatenate(p)
        per_exit = {}
        for j, n in enumerate(EXIT_POSITIONS):
            m = ds.num_packet >= n
            yt = ds.labels[m]
            pj = {k: v[m, j] for k, v in preds.items()}
            per_exit[str(n)] = {
                "n": int(m.sum()),
                "macro_f1": {k: float(f1_score(yt, v, average="macro", zero_division=0)) for k, v in pj.items()},
                "n_disagree": {
                    "fp32_full_vs_fp32_incremental": int((pj["fp32_full"] != pj["fp32_incremental"]).sum()),
                    "bf16_full_vs_bf16_incremental": int((pj["bf16_full"] != pj["bf16_incremental"]).sum()),
                    "fp32_incremental_vs_bf16_incremental": int((pj["fp32_incremental"] != pj["bf16_incremental"]).sum()),
                },
            }
            print(f"seed={seed} exit={n} {json.dumps(per_exit[str(n)])}", flush=True)
        results[str(seed)] = per_exit
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


def bench(fn, iters, repeats=5, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / iters)
    return float(np.median(times)) * 1000  # ms


def nparams_ckpt(path):
    return sum(v.numel() for v in torch.load(path, map_location="cpu").values())


def part_timing(out_path, quick):
    nb_classes = get_nb_classes(DATASET)
    ds = ByteDatasetMultiExit(f"{DATA_ROOT}/{DATASET}/data-test.json")
    x_all = torch.from_numpy(ds.bytes[ds.num_packet >= NUM_PACKET])
    g = torch.Generator().manual_seed(0)
    model = load_fme(SEEDS[0], nb_classes, torch.bfloat16)
    singles = [load_single_n(n, SEEDS[0], nb_classes, torch.bfloat16) for n in EXIT_POSITIONS]
    res = {"dataset": DATASET, "checkpoint_seed": SEEDS[0], "dtype": "bf16", "gpu": torch.cuda.get_device_name(0),
           "input": f"real test flows with {NUM_PACKET} packets, fixed random sample"}
    with torch.no_grad():
        for bs in [1, 128]:
            x = x_all[torch.randperm(len(x_all), generator=g)[:bs]].to(DEVICE)
            iters = 2 if quick else (200 if bs == 1 else 20)
            r = {"naive_exit_ms": {}, "incremental_packet_update_ms": {}}
            for j, n in enumerate(EXIT_POSITIONS):
                r["naive_exit_ms"][str(n)] = bench(lambda n=n, j=j: readout(model, j, backbone(model, x[:, :n * NUM_PACKET_BYTE])[:, -1]), iters)
            for k in range(NUM_PACKET):
                st0 = IncrementalState(len(model.layers))
                for kk in range(k):
                    model_forward_incremental_chunk(model, chunk(x, kk), st0)

                def step(st0=st0, k=k):
                    # fresh lists pointing at the prebuilt state tensors; the update replaces them rather than writing in place
                    st = IncrementalState(len(model.layers))
                    st.conv, st.ssm = list(st0.conv), list(st0.ssm)
                    h_last = model_forward_incremental_chunk(model, chunk(x, k), st)
                    if k + 1 in EXIT_POSITIONS:
                        readout(model, EXIT_POSITIONS.index(k + 1), h_last)
                r["incremental_packet_update_ms"][str(k + 1)] = bench(step, iters)
            r["cumulative_all_exits_ms"] = {
                "naive_recompute": bench(lambda: naive_exits(model, x), iters),
                "shared_incremental": bench(lambda: incremental_exits(model, x), iters),
                "specialists_incremental": bench(lambda: specialists_incremental_exits(singles, x), iters),
                "specialists_recompute": bench(lambda: specialists_recompute_exits(singles, x), iters),
            }
            r["flows_per_s_all_exits"] = {k: bs / (v / 1000) for k, v in r["cumulative_all_exits_ms"].items()}
            res[f"bs{bs}"] = r
            print(f"bs={bs} {json.dumps(r)}", flush=True)

        st = IncrementalState(len(model.layers))
        model_forward_incremental_chunk(model, chunk(x_all[:1].to(DEVICE), 0), st)
        per_flow = sum(t.numel() * t.element_size() for t in st.conv + st.ssm)
        res["state"] = {
            "bytes_per_flow": per_flow, "conv_state_dtype": str(st.conv[0].dtype), "ssm_state_dtype": str(st.ssm[0].dtype),
            "gb_per_10k_concurrent_flows": per_flow * 1e4 / 1e9,
            "recompute_raw_byte_buffer_per_flow": NUM_PACKET * NUM_PACKET_BYTE,
        }
    res["params"] = {
        "flowmamba_early": nparams_ckpt(f"{RUNS}/flowmamba_early_{DATASET}_seed{SEEDS[0]}/checkpoint-best.pth"),
        "flowmamba_single_n_x4": sum(nparams_ckpt(f"{RUNS}/flowmamba_single_n{n}_{DATASET}_seed{SEEDS[0]}/checkpoint-best.pth")
                                     for n in EXIT_POSITIONS),
        "netmamba_n_x4": sum(nparams_ckpt(f"{RUNS}/netmamba_n{n}_{DATASET}_seed{SEEDS[0]}/checkpoint-best.pth") for n in EXIT_POSITIONS),
        "mambanetburst_n_x4": sum(nparams_ckpt(f"{RUNS}/mambanetburst_n{n}_{DATASET}_seed{SEEDS[0]}/checkpoint-best.pth") for n in EXIT_POSITIONS),
    }
    print(json.dumps({"state": res["state"], "params": res["params"]}), flush=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["accuracy", "timing"], required=True)
    ap.add_argument("--out", default=None, help="output JSON; defaults to OUTPUT_ROOT/results/incremental_agreement.json "
                                                "or OUTPUT_ROOT/results/deployment_benchmark.json")
    ap.add_argument("--quick", action="store_true", help="two timing iterations only, for a code smoke test")
    args = ap.parse_args()
    os.makedirs(RESULTS_OUT, exist_ok=True)
    if args.part == "accuracy":
        part_accuracy(args.out or f"{RESULTS_OUT}/incremental_agreement.json")
    else:
        part_timing(args.out or f"{RESULTS_OUT}/deployment_benchmark.json", args.quick)


if __name__ == "__main__":
    main()
