"""Statistics of the controlled comparison, computed from OUTPUT_ROOT/results/per_seed.csv and
the per-flow predictions of the experiment runs. Writes OUTPUT_ROOT/results/analysis/summary.json,
one CSV per comparison, per_class.json and confusion.json. All selection uses validation
scores; test scores are only reported.

Main comparison: FME with the common recipe against the conservative single-exit baseline,
which is, for each dataset and exit, whichever of Single-orig and Single-stepmatch has the
higher mean validation macro-F1 over the seeds. For each dataset-exit pair the paired seeds
give the mean difference with a t-based 95% confidence interval, a paired t-test and an exact
Wilcoxon signed-rank test; Holm's correction runs over the 18 comparable pairs, and a Wilcoxon
test over the 18 mean differences summarizes the direction. A test-flow bootstrap resamples
connections, both directions of a connection together and flows without parsed endpoints as
their own clusters, and averages over seeds.

Further outputs: the subsets of test_common, test_unseen_conn and test_unseen_file; the long
budget (long50*); test-time masks (testmask_*); retraining with the combined mask (masked*);
per-exit checkpoint selection (per_exit_checkpoint*); the eligibility-matched model
(eligibility_matched*); per-class F1, pooled confusion matrices and decisions across exits.

    python src/analysis.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import experiment as L
from paths import RESULTS_OUT

OUT = f"{RESULTS_OUT}/analysis"
SEEDS5 = [20260921, 20260922, 20260923, 20260924, 20260925]
SEEDS10 = SEEDS5 + [20260926, 20260927, 20260928, 20260929, 20260930]
SEEDS3 = SEEDS5[:3]
EXCLUDED = {("CipherSpectrum", 1), ("CipherSpectrum", 2)}
DS = [L.NAMES[d] for d in L.DATASETS]
KEY = {v: k for k, v in L.NAMES.items()}


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def scores(df, model, variant, dataset, exit_, eval_set, seeds):
    s = df[(df.model == model) & (df.variant == variant) & (df.dataset == dataset) & (df.exit == exit_) & (df.eval_set == eval_set)]
    s = s.set_index("seed").macro_f1
    return np.array([s.get(x, np.nan) for x in seeds], float)


def has(df, model, variant, dataset, exit_):
    return ((df.model == model) & (df.variant == variant) & (df.dataset == dataset) & (df.exit == exit_)).any()


def conservative(df, dataset, exit_, seeds, variants=("orig", "stepmatch")):
    """Single-orig or Single-stepmatch (or their masked versions), whichever has the higher mean validation macro-F1."""
    cands = [v for v in variants if has(df, "Single", v, dataset, exit_)]
    val = {v: np.nanmean(scores(df, "Single", v, dataset, exit_, "val", seeds)) for v in cands}
    return max(val, key=val.get), val


def pair_stats(a, b):
    d = a - b
    ok = ~np.isnan(d)
    d = d[ok]
    n = len(d)
    out = {"n_seeds": int(n), "a_mean": float(np.nanmean(a)), "a_sd": float(np.nanstd(a, ddof=1)),
           "b_mean": float(np.nanmean(b)), "b_sd": float(np.nanstd(b, ddof=1)), "diff_mean": float(d.mean()) if n else np.nan}
    if n >= 2:
        se = d.std(ddof=1) / np.sqrt(n)
        h = stats.t.ppf(0.975, n - 1) * se
        out.update(ci_low=float(d.mean() - h), ci_high=float(d.mean() + h),
                   t_p=float(stats.ttest_rel(a[ok], b[ok]).pvalue) if se > 0 else (0.0 if d.mean() != 0 else 1.0))
        try:
            out["wilcoxon_p"] = float(stats.wilcoxon(d, method="exact").pvalue)
        except ValueError:
            out["wilcoxon_p"] = 1.0
    out["clear"] = bool(abs(out["diff_mean"]) > max(out["a_sd"], out["b_sd"]))
    out["separated"] = bool(np.nanmin(a) > np.nanmax(b)) if out["diff_mean"] > 0 else bool(np.nanmax(a) < np.nanmin(b))
    return out


def comparison(df, a, b, eval_set, seeds, datasets=DS, exits=L.EXITS, b_rule=None, variants=("orig", "stepmatch")):
    """a, b: (model, variant); b_rule='conservative' picks the single-exit baseline per pair among `variants`."""
    rows = []
    for ds in datasets:
        for e in exits:
            if (ds, e) in EXCLUDED or not has(df, *a, ds, e):
                continue
            bv, val = (conservative(df, ds, e, seeds, variants) if b_rule == "conservative" else (b[1], None))
            if not has(df, b[0], bv, ds, e):
                continue
            r = pair_stats(scores(df, *a, ds, e, eval_set, seeds), scores(df, b[0], bv, ds, e, eval_set, seeds))
            r.update(dataset=ds, exit=e, baseline_variant=bv, baseline_val_means=val,
                     n_flows=int(df[(df.model == a[0]) & (df.variant == a[1]) & (df.dataset == ds) & (df.exit == e) & (df.eval_set == eval_set)].n_flows.max()))
            rows.append(r)
    if rows and "t_p" in rows[0]:
        adj = holm([r.get("t_p", 1.0) for r in rows])
        for r, p in zip(rows, adj):
            r["t_p_holm"] = float(p)
    return rows


def overall(rows):
    d = np.array([r["diff_mean"] for r in rows])
    w = stats.wilcoxon(d, method="exact") if len(d) <= 25 else stats.wilcoxon(d)
    return {"pairs": len(d), "positive": int((d > 0).sum()), "negative": int((d < 0).sum()),
            "clear_positive": sum(r["clear"] and r["diff_mean"] > 0 for r in rows),
            "clear_negative": sum(r["clear"] and r["diff_mean"] < 0 for r in rows),
            "separated_positive": sum(r["separated"] and r["diff_mean"] > 0 for r in rows),
            "holm_significant_positive": sum(r.get("t_p_holm", 1) < 0.05 and r["diff_mean"] > 0 for r in rows),
            "holm_significant_negative": sum(r.get("t_p_holm", 1) < 0.05 and r["diff_mean"] < 0 for r in rows),
            "median_diff": float(np.median(d)), "mean_diff": float(d.mean()), "wilcoxon_p": float(w.pvalue)}


def run_dirs(model, variant, dataset_key, seed, n=None):
    job = {"model": model, "variant": variant, "dataset": dataset_key, "seed": seed, "n": n}
    return L.run_dir_for(job)[0]


def test_preds(model, variant, ds, e, seed):
    key = KEY[ds]
    z = np.load(f"{run_dirs(model, variant, key, seed, None if model == 'FME' else e)}/preds_best.npz")
    return z[f"test_e{e}_index"], z[f"test_e{e}_y"].astype(int), z[f"test_e{e}_pred"].astype(int)


def wf1(y, p, w, k):
    tp = np.bincount(y[y == p], weights=w[y == p], minlength=k)
    pos_t = np.bincount(y, weights=w, minlength=k)
    pos_p = np.bincount(p, weights=w, minlength=k)
    present = (pos_t > 0) | (pos_p > 0)
    f = 2 * tp[present] / np.maximum(pos_t[present] + pos_p[present], 1e-12)
    return f.mean()


def cluster_bootstrap(df, rows, a, seeds, reps=1000, rng_seed=20261105):
    rng = np.random.default_rng(rng_seed)
    for r in rows:
        ds, e = r["dataset"], r["exit"]
        meta = pd.read_csv(f"{L.OUT}/meta/{KEY[ds]}_test.csv").set_index("index")
        try:
            A = [test_preds(*a, ds, e, s) for s in seeds]
            B = [test_preds("Single", r["baseline_variant"], ds, e, s) for s in seeds]
        except FileNotFoundError as exc:
            r["cluster_boot_ci"], r["cluster_boot_missing"] = None, str(exc)
            continue
        idx = A[0][0]
        assert all((x[0] == idx).all() for x in A + B)
        conn = meta.loc[idx, "connection"].fillna("").to_numpy()
        keys = np.where(conn == "", np.array([f"flow{i}" for i in idx]), conn)
        _, cl = np.unique(keys, return_inverse=True)
        k = int(max(x[1].max() for x in A + B) + 1)
        nclu = cl.max() + 1
        diffs = []
        for _ in range(reps):
            draw = np.bincount(rng.integers(0, nclu, nclu), minlength=nclu)
            w = draw[cl].astype(float)
            fa = np.mean([wf1(y, p, w, k) for _, y, p in A])
            fb = np.mean([wf1(y, p, w, k) for _, y, p in B])
            diffs.append(fa - fb)
        r["cluster_boot_ci"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
        r["clusters"] = int(nclu)


def mf1(y, p):
    return L.macro_f1(np.asarray(y), np.asarray(p))


def epoch_scores(rd, e):
    """Validation and test macro-F1 of exit e at every epoch of a run with preds_epochs.npz."""
    z = np.load(f"{rd}/preds_epochs.npz")
    yv, pv, yt, pt = z[f"valid_e{e}_y"], z[f"valid_e{e}_pred"], z[f"test_e{e}_y"], z[f"test_e{e}_pred"]
    return [mf1(yv, pv[k]) for k in range(len(pv))], [mf1(yt, pt[k]) for k in range(len(pt))]


def per_exit_checkpoint():
    """Per-exit checkpoint selection against the mean-over-exits rule, within the same FME runs."""
    rows = []
    for variant, seeds in [("orig", SEEDS10[3:])]:
        for d in L.DATASETS:
            for s in seeds:
                rd = run_dirs("FME", variant, d, s)
                if not os.path.exists(f"{rd}/preds_epochs.npz") or not os.path.exists(f"{rd}/result.json"):
                    continue
                cfg, res = json.load(open(f"{rd}/config.json")), json.load(open(f"{rd}/result.json"))
                be = cfg["best_epoch"]
                for e in L.EXITS:
                    if (L.NAMES[d], e) in EXCLUDED:
                        continue
                    vf, tf = epoch_scores(rd, e)
                    assert abs(tf[be] - res["test"][str(e)]["f1"]) < 1e-9, (rd, e)
                    ke = int(np.argmax(vf))
                    rows.append({"dataset": L.NAMES[d], "exit": e, "seed": s, "variant": variant, "mean_rule_epoch": be,
                                 "exit_rule_epoch": ke, "test_mean_rule": tf[be], "test_exit_rule": tf[ke]})
    df = pd.DataFrame(rows)
    if df.empty:
        return [], []
    df["diff"] = df.test_exit_rule - df.test_mean_rule
    df["same_epoch"] = (df.mean_rule_epoch == df.exit_rule_epoch).astype(int)
    g = df.groupby(["dataset", "exit"]).agg(runs=("diff", "size"), mean_diff=("diff", "mean"), sd_diff=("diff", "std"),
                                            same_epoch=("same_epoch", "sum")).reset_index()
    return df.to_dict("records"), g.to_dict("records")


def eligibility_matched(df):
    """Eligibility-matched FME against the step-matched single-exit model and the original FME."""
    rows = []
    for d in ["iscxvpn", "ciciot2022", "ustctfc2016"]:
        ds = L.NAMES[d]
        for n in [2, 3, 5]:
            for s in SEEDS5:
                rd = run_dirs("FME", f"elig{n}", d, s, n)
                if not os.path.exists(f"{rd}/result.json"):
                    continue
                cfg = json.load(open(f"{rd}/config.json"))
                vf, tf = epoch_scores(rd, n)
                rows.append({"dataset": ds, "exit": n, "seed": s, "elig_mean_rule": tf[cfg["best_epoch"]],
                             "elig_exit_rule": tf[int(np.argmax(vf))],
                             "fme_orig": scores(df, "FME", "orig", ds, n, "test", [s])[0],
                             "single_orig": scores(df, "Single", "orig", ds, n, "test", [s])[0],
                             "single_stepmatch": scores(df, "Single", "stepmatch", ds, n, "test", [s])[0]})
    t = pd.DataFrame(rows)
    if t.empty:
        return [], []
    t["objective"] = t.elig_exit_rule - t.single_stepmatch
    t["shorter_flows"] = t.fme_orig - t.elig_mean_rule
    t["total_vs_stepmatch"] = t.fme_orig - t.single_stepmatch
    g = t.groupby(["dataset", "exit"]).agg(seeds=("seed", "size"), **{c: (c, "mean") for c in
        ["fme_orig", "elig_mean_rule", "elig_exit_rule", "single_stepmatch", "single_orig",
         "objective", "shorter_flows", "total_vs_stepmatch"]},
        objective_sd=("objective", "std"), shorter_flows_sd=("shorter_flows", "std")).reset_index()
    # Paired statistics of the two single-factor contrasts: the multi-exit objective on the same flows and steps
    # (eligibility-matched FME selected at exit N against step-matched Single-N), and the shorter flows under the
    # same selection rule (original FME against eligibility-matched FME, both selected by the mean over exits).
    # Holm's correction runs over the nine pairs of each contrast.
    extra = []
    for (ds, n), x in t.groupby(["dataset", "exit"]):
        r = {"dataset": ds, "exit": n}
        for name, a, b in [("objective", "elig_exit_rule", "single_stepmatch"), ("shorter_flows", "fme_orig", "elig_mean_rule")]:
            st = pair_stats(x[a].to_numpy(float), x[b].to_numpy(float))
            r.update({f"{name}_{k}": st.get(k, np.nan) for k in ["diff_mean", "ci_low", "ci_high", "t_p", "clear", "separated"]})
        extra.append(r)
    for name in ["objective", "shorter_flows"]:
        for r, p in zip(extra, holm([r[f"{name}_t_p"] for r in extra])):
            r[f"{name}_t_p_holm"] = float(p)
    g = g.merge(pd.DataFrame(extra), on=["dataset", "exit"])
    return t.to_dict("records"), g.to_dict("records")


def pooled_predictions(model, variant, ds, e, seeds):
    out = [test_preds(model, variant, ds, e, s) for s in seeds]
    return out


def per_class(df, rows, seeds, exits=(3, 5)):
    """Per-class F1 averaged over seeds, FME against the conservative baseline of each pair."""
    res = {}
    for r in rows:
        ds, e = r["dataset"], r["exit"]
        if e not in exits:
            continue
        A = pooled_predictions("FME", "orig", ds, e, seeds)
        B = pooled_predictions("Single", r["baseline_variant"], ds, e, seeds)
        classes = np.unique(A[0][1])
        fa = np.mean([stats_f1(y, p, classes) for _, y, p in A], axis=0)
        fb = np.mean([stats_f1(y, p, classes) for _, y, p in B], axis=0)
        d = 100 * (fa - fb)
        res[f"{ds}|{e}"] = {"classes": classes.tolist(), "fme": fa.tolist(), "baseline": fb.tolist(), "diff_points": d.tolist(),
                            "up": int((d > 0.5).sum()), "down": int((d < -0.5).sum()), "median": float(np.median(d)),
                            "mean": float(d.mean()), "baseline_variant": r["baseline_variant"]}
    return res


def stats_f1(y, p, classes):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=classes, average=None, zero_division=0)


def confusion(rows, seeds, datasets=("ISCXVPN2016", "CICIoT2022"), e=5):
    res = {}
    for r in rows:
        if r["dataset"] not in datasets or r["exit"] != e:
            continue
        for name, (model, variant) in {"fme": ("FME", "orig"), "baseline": ("Single", r["baseline_variant"])}.items():
            P = pooled_predictions(model, variant, r["dataset"], e, seeds)
            k = int(max(max(y.max(), p.max()) for _, y, p in P) + 1)
            cm = np.zeros((k, k), int)
            for _, y, p in P:
                np.add.at(cm, (y, p), 1)
            res[f"{r['dataset']}|{name}"] = {"counts": cm.tolist(), "variant": variant}
    return res


def stability(rows, seeds):
    """Decisions at an earlier exit and at exit 5 on the test flows with at least five packets."""
    base = {(r["dataset"], r["exit"]): r["baseline_variant"] for r in rows}
    out = []
    for ds in DS:
        for first in [1, 3]:
            if (ds, first) in EXCLUDED or (ds, 5) not in base or (ds, first) not in base:
                continue
            vals = {"FME": [], "Single": []}
            for s in seeds:
                for model in ["FME", "Single"]:
                    v1 = "orig" if model == "FME" else base[(ds, first)]
                    v5 = "orig" if model == "FME" else base[(ds, 5)]
                    i1, y1, p1 = test_preds(model, v1, ds, first, s)
                    i5, y5, p5 = test_preds(model, v5, ds, 5, s)
                    m = dict(zip(i1, range(len(i1))))
                    keep = np.array([m[i] for i in i5])
                    a, b, y = p1[keep], p5, y5
                    c1, c5 = a == y, b == y
                    vals[model].append([100 * (a == b).mean(), 100 * (c1 & ~c5).sum() / max(c1.sum(), 1),
                                        100 * (~c1 & c5).sum() / max((~c1).sum(), 1)])
            row = {"dataset": ds, "exits": f"{first} to 5", "flows": int(len(i5))}
            for model in vals:
                v = np.mean(vals[model], axis=0)
                row.update({f"{model}_same": v[0], f"{model}_correct_to_wrong": v[1], f"{model}_wrong_to_correct": v[2]})
            out.append(row)
    return out


def masked_retraining(df):
    """Retraining with mask_all on training, validation and test inputs, on every dataset that has it.
    Seeds are those whose masked runs of the dataset have all finished. ISCXVPN2016 also has a masked step-matched
    single-exit variant, so its baseline is chosen conservatively by validation score as in the main
    comparison; the other datasets use the masked original recipe. Holm's correction runs over all
    masked pairs together. The unmasked margins use the same seeds and the same baseline rule."""
    out = {k: [] for k in ["masked", "masked_unmasked_same_seeds", "masked_unseen_conn", "masked_cost_fme", "masked_cost_single"]}
    for ds in DS:
        if not any(has(df, "FME", "maskall", ds, e) for e in L.EXITS):
            continue
        cons = has(df, "Single", "maskall_stepmatch", ds, 2)
        exits = [e for e in L.EXITS if (ds, e) not in EXCLUDED]

        def complete(s):
            # a seed counts only when every masked model it needs has finished
            need = [("FME", "maskall", e) for e in exits] + [("Single", "maskall", e) for e in exits]
            need += [("Single", "maskall_stepmatch", e) for e in exits if cons and e > 1]
            return all(((df.model == m) & (df.variant == v) & (df.dataset == ds) & (df.exit == e) & (df.seed == s)
                        & (df.eval_set == "test")).any() for m, v, e in need)
        seeds = [s for s in SEEDS10 if complete(s)]
        n_before = {k: len(v) for k, v in out.items()}
        kw = dict(b_rule="conservative", variants=("maskall", "maskall_stepmatch")) if cons else {}
        out["masked"] += comparison(df, ("FME", "maskall"), ("Single", "maskall"), "test", seeds, datasets=[ds], **kw)
        out["masked_unseen_conn"] += comparison(df, ("FME", "maskall"), ("Single", "maskall"), "test_unseen_conn", seeds, datasets=[ds], **kw)
        if cons:
            out["masked_unmasked_same_seeds"] += comparison(df, ("FME", "orig"), ("Single", None), "test", seeds, datasets=[ds], b_rule="conservative")
        else:
            out["masked_unmasked_same_seeds"] += comparison(df, ("FME", "orig"), ("Single", "orig"), "test", seeds, datasets=[ds])
        out["masked_cost_fme"] += comparison(df, ("FME", "orig"), ("FME", "maskall"), "test", seeds, datasets=[ds])
        out["masked_cost_single"] += comparison(df, ("Single", "orig"), ("Single", "maskall"), "test", seeds, datasets=[ds])
        for k, v in out.items():
            for r in v[n_before[k]:]:
                r["seeds"] = json.dumps(seeds)
    for k in ["masked", "masked_unseen_conn"]:
        rows = out[k]
        if rows and all("t_p" in r for r in rows):
            for r, p in zip(rows, holm([r["t_p"] for r in rows])):
                r["t_p_holm"] = float(p)
    if out["masked"]:
        out["masked_overall"] = overall(out["masked"])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(f"{RESULTS_OUT}/per_seed.csv")
    df = df[df.status == "ok"]
    res = {}
    seeds = [s for s in SEEDS10 if ((df.model == "FME") & (df.variant == "orig") & (df.seed == s)).any()
             and ((df.model == "Single") & (df.variant == "orig") & (df.seed == s)).any()]
    res["seeds_main"] = seeds
    main_rows = comparison(df, ("FME", "orig"), ("Single", None), "test", seeds, b_rule="conservative")
    if all(has(df, "FME", "orig", r["dataset"], r["exit"]) for r in main_rows):
        cluster_bootstrap(df, main_rows, ("FME", "orig"), seeds)
    res["main_vs_conservative"] = main_rows
    res["main_vs_conservative_overall"] = overall(main_rows)
    for name, b in [("vs_single_orig", ("Single", "orig")), ("vs_single_stepmatch", ("Single", "stepmatch"))]:
        rows = comparison(df, ("FME", "orig"), b, "test", seeds)
        res[f"main_{name}"] = rows
        res[f"main_{name}_overall"] = overall(rows) if rows else None
    for subset in ["test_common", "test_unseen_conn", "test_unseen_file"]:
        rows = [r for r in comparison(df, ("FME", "orig"), ("Single", None), subset, seeds, b_rule="conservative") if r["n_flows"] >= 50]
        res[subset] = rows
        res[f"{subset}_overall"] = overall(rows) if rows else None
    long_ds = ["ISCXVPN2016", "CSTNET-TLS1.3"]
    if has(df, "FME", "long50", "ISCXVPN2016", 1):
        rows = comparison(df, ("FME", "long50"), ("Single", "long50"), "test", SEEDS5, datasets=long_ds)
        res["long50"] = rows
        res["long50_overall"] = overall(rows)
        res["long50_vs_orig_fme"] = comparison(df, ("FME", "long50"), ("FME", "orig"), "test", SEEDS5, datasets=long_ds)
        res["long50_vs_orig_single"] = comparison(df, ("Single", "long50"), ("Single", "orig"), "test", SEEDS5, datasets=long_ds)
    for mask in ["mask_sni", "mask_hdr", "mask_trailer", "mask_both", "mask_all"]:
        v = f"orig_at_{mask}"
        if has(df, "FME", v, "CSTNET-TLS1.3", 1):
            rows = comparison(df, ("FME", v), ("Single", v), "test", SEEDS3)
            res[f"testmask_{mask}"] = rows
            res[f"testmask_{mask}_overall"] = overall(rows)
    res.update(masked_retraining(df))
    res["per_exit_checkpoint_runs"], res["per_exit_checkpoint"] = per_exit_checkpoint()
    res["eligibility_matched_runs"], res["eligibility_matched"] = eligibility_matched(df)
    complete = all(len(scores(df, "FME", "orig", r["dataset"], r["exit"], "test", seeds)) == len(seeds) for r in main_rows)
    if complete and len(seeds) >= 5:
        json.dump(per_class(df, main_rows, seeds), open(f"{OUT}/per_class.json", "w"), indent=1)
        json.dump(confusion(main_rows, seeds), open(f"{OUT}/confusion.json", "w"))
        res["stability"] = stability(main_rows, seeds)
    json.dump(res, open(f"{OUT}/summary.json", "w"), indent=2, default=float)
    for k, v in res.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            pd.DataFrame(v).to_csv(f"{OUT}/{k}.csv", index=False)
    print(json.dumps({k: v for k, v in res.items() if k.endswith("overall") or k == "seeds_main"}, indent=1, default=float))


if __name__ == "__main__":
    main()
