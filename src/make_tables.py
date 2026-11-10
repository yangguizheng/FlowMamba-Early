"""Rows of the LaTeX tables of the paper and its supporting information, written from the result
files so that no number is typed by hand.

Reads RESULTS_IN (OUTPUT_ROOT/results by default; FLOWMAMBA_RESULTS=results reads the result files
published with this repository) and writes one file of table rows per table to OUTPUT_ROOT/tables.
File names start with the table they fill: tab_* for the main text and S<k>_* for S<k> Appendix.

    python src/make_tables.py
"""
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from paths import OUTPUT_ROOT, RESULTS_IN

AN = f"{RESULTS_IN}/analysis"
OUT = f"{OUTPUT_ROOT}/tables"
os.makedirs(OUT, exist_ok=True)
DS = ["ISCXVPN2016", "CICIoT2022", "USTC-TFC2016", "CipherSpectrum", "CSTNET-TLS1.3"]
KEY = {"ISCXVPN2016": "iscxvpn", "CICIoT2022": "ciciot2022", "USTC-TFC2016": "ustctfc2016",
       "CipherSpectrum": "cipherspectrum", "CSTNET-TLS1.3": "cstnet-tls1.3"}
CLASSES = {"ISCXVPN2016": 7, "CICIoT2022": 6, "USTC-TFC2016": 20, "CipherSpectrum": 41, "CSTNET-TLS1.3": 120}
EXITS = [1, 2, 3, 5]
LOW = {("CipherSpectrum", 1), ("CipherSpectrum", 2)}
SEEDS = list(range(20260921, 20260931))
SEEDS5 = SEEDS[:5]
SEEDS3 = SEEDS[:3]
DAG = r"\textsuperscript{\dag}"

ps = pd.read_csv(f"{RESULTS_IN}/per_seed.csv")
ps = ps[ps.status == "ok"]


def scores(model, variant, ds, n, ev="test", seeds=SEEDS):
    s = ps[(ps.model == model) & (ps.variant == variant) & (ps.dataset == ds) & (ps.exit == n) & (ps.eval_set == ev)]
    s = s.set_index("seed").macro_f1
    return np.array([s[x] for x in seeds if x in s.index])


def stat_table(name):
    t = pd.read_csv(f"{AN}/{name}.csv")
    return {(r.dataset, int(r.exit)): r for r in t.itertuples()}


def ms(a):
    return f"${a.mean():.3f}\\pm{a.std(ddof=1):.3f}$"


def margin(r, holm=True):
    v = f"{100 * r.diff_mean:+.1f}".replace("-", "-")
    star = "^{*}" if holm and getattr(r, "t_p_holm", 1.0) < 0.05 else ""
    body = f"\\boldsymbol{{{v}}}" if r.clear else v
    return f"${body}{star}$"


def pval(p):
    if p < 0.001:
        e = int(np.floor(np.log10(p)))
        return f"${p / 10 ** e:.1f}\\times10^{{{e}}}$"
    return f"{p:.3f}"


def sg(x, d=1):
    return f"${x:+.{d}f}$"


def ci(lo, hi):
    return f"$[{lo:.1f}, {hi:.1f}]$"


def write(name, text):
    with open(f"{OUT}/{name}.tex", "w") as f:
        f.write(text + "\n")
    print("wrote", name)


def comparison_block(stats, ev, counts=None, classes=True):
    """Rows FME / Single-N / Margin for each dataset; Single-N cells of step-matched baselines carry a dagger."""
    lines = []
    for i, ds in enumerate(DS):
        fme, sing, mar = [], [], []
        for n in EXITS:
            r = stats.get((ds, n))
            variant = r.baseline_variant if r is not None else "orig"
            fme.append(ms(scores("FME", "orig", ds, n, ev)))
            cell = ms(scores("Single", variant, ds, n, ev))
            sing.append(cell + (DAG if variant == "stepmatch" else ""))
            mar.append("--" if r is None else margin(r))
        label = f"{ds} ({CLASSES[ds]} classes)" if classes else ds
        if counts is not None:
            lines.append(f"{ds} & Flows & " + " & ".join(f"{counts[(ds, n)]:,}" for n in EXITS) + r" \\")
            label = ""
        lines.append(f"{label} & FME & " + " & ".join(fme) + r" \\")
        lines.append(" & Single-N & " + " & ".join(sing) + r" \\")
        lines.append(" & Margin & " + " & ".join(mar) + r" \\")
        if i < len(DS) - 1:
            lines.append(r"\midrule")
    return "\n".join(lines)


# ---------------------------------------------------------------- main text: Table of the main comparison
main = stat_table("main_vs_conservative")
write("tab_main", comparison_block(main, "test"))

# ---------------------------------------------------------------- main text: decisions across exits
st = pd.read_csv(f"{AN}/stability.csv")
lines = []
prev = None
for r in st.itertuples():
    if prev is not None and r.dataset != prev:
        lines.append(r"\midrule")
    first = r.dataset != prev
    prev = r.dataset
    lines.append(f"{r.dataset if first else ''} & {r.exits} & {r.FME_same:.1f} & {r.Single_same:.1f} & {r.FME_correct_to_wrong:.1f} & "
                 f"{r.Single_correct_to_wrong:.1f} & {r.FME_wrong_to_correct:.1f} & {r.Single_wrong_to_correct:.1f} \\\\")
write("tab_agree", "\n".join(lines))

# ---------------------------------------------------------------- S1: common population
common = stat_table("test_common")
write("S1_common", comparison_block(common, "test_common", classes=False))

# ---------------------------------------------------------------- S2: unseen connections and unseen capture files
unseen = stat_table("test_unseen_conn")
counts = {(r.dataset, int(r.exit)): int(r.n_flows) for r in unseen.values()}
for ds in DS:
    for n in EXITS:
        if (ds, n) not in counts:
            counts[(ds, n)] = int(ps[(ps.model == "FME") & (ps.variant == "orig") & (ps.dataset == ds) & (ps.exit == n)
                                     & (ps.eval_set == "test_unseen_conn")].n_flows.iloc[0])
write("S2_unseen_conn", comparison_block(unseen, "test_unseen_conn", counts=counts))
uf = stat_table("test_unseen_file")
lines = []
for ds in ["CipherSpectrum", "CSTNET-TLS1.3"]:
    for n in EXITS:
        r = uf.get((ds, n))
        if r is None:
            continue
        a = scores("FME", "orig", ds, n, "test_unseen_file")
        b = scores("Single", r.baseline_variant, ds, n, "test_unseen_file")
        lines.append(f"{ds if n == min(k[1] for k in uf if k[0] == ds) else ''} & {n} & {int(r.n_flows):,} & {ms(a)} & {ms(b)} & {margin(r, holm=False)} & {pval(r.t_p)} \\\\")
    if ds == "CipherSpectrum":
        lines.append(r"\midrule")
write("S2_unseen_file", "\n".join(lines))

# ---------------------------------------------------------------- S3: every seed, and the paired statistics
for model, variant, name, dsets in [("FME", "orig", "S3_fme_seeds", DS), ("Single", "orig", "S3_single_orig_seeds", DS),
                                    ("Single", "stepmatch", "S3_single_step_seeds", DS[:3])]:
    lines = []
    for i, ds in enumerate(dsets):
        ex = EXITS if variant == "orig" else [2, 3, 5]
        for n in ex:
            v = scores(model, variant, ds, n)
            assert len(v) == 10, (model, variant, ds, n)
            lines.append(f"{ds if n == ex[0] else ''} & {n} & " + " & ".join(f"{x:.3f}" for x in v) + r" \\")
        if i < len(dsets) - 1:
            lines.append(r"\midrule")
    write(name, "\n".join(lines))

so = stat_table("main_vs_single_orig")
ss = stat_table("main_vs_single_stepmatch")
lines, lines2 = [], []
prev = None
for ds in DS:
    for n in EXITS:
        if (ds, n) in LOW:
            continue
        r = main[(ds, n)]
        if prev is not None and ds != prev:
            lines.append(r"\midrule")
            lines2.append(r"\midrule")
        lo, hi = json.loads(r.cluster_boot_ci)
        o = so[(ds, n)]
        s = ss.get((ds, n))
        step = "-- & --" if s is None else f"{sg(100 * s.diff_mean)} & {pval(s.t_p_holm)}"
        lines.append(f"{ds if ds != prev else ''} & {n} & {'step-matched' if r.baseline_variant == 'stepmatch' else 'original'} & "
                     f"{sg(100 * r.diff_mean)} & {ci(100 * r.ci_low, 100 * r.ci_high)} & {pval(r.t_p)} & {pval(r.wilcoxon_p)} & "
                     f"{pval(r.t_p_holm)} & {ci(100 * lo, 100 * hi)} & {'yes' if r.separated else 'no'} \\\\")
        lines2.append(f"{ds if ds != prev else ''} & {n} & {sg(100 * o.diff_mean)} & {pval(o.t_p_holm)} & {step} \\\\")
        prev = ds
write("S3_stats", "\n".join(lines))
write("S3_stats_variants", "\n".join(lines2))

# ---------------------------------------------------------------- S6: server names, test-time masks, masked retraining
sni = json.load(open(f"{RESULTS_IN}/sni_audit.json"))
lines = []
for i, ds in enumerate(DS):
    d = sni[KEY[ds]]
    hv = d["host_vs_label"]
    named = sum(hv.values())
    in_host = hv.get("label_in_host", 0)
    first = d["first_sni_packet"]
    for n in EXITS:
        e = d["sni_by_exit"][str(n)]
        lines.append(f"{ds if n == 1 else ''} & {n} & {e['eligible_flows']:,} & {100 * e['share_with_sni_in_first_n']:.1f} & "
                     + (f"{named:,} & {in_host:,}" if n == 1 else " & ") + r" \\")
    if i < len(DS) - 1:
        lines.append(r"\midrule")
write("S6_sni", "\n".join(lines))

MASKS = [("orig", "none"), ("orig_at_mask_sni", "SNI"), ("orig_at_mask_hdr", "header"), ("orig_at_mask_trailer", "trailer"),
         ("orig_at_mask_all", "all")]
lines = []
for i, ds in enumerate(DS):
    for n in EXITS:
        if (ds, n) in LOW:
            continue
        f = [100 * scores("FME", v, ds, n, seeds=SEEDS3).mean() for v, _ in MASKS]
        s = [100 * scores("Single", v, ds, n, seeds=SEEDS3).mean() for v, _ in MASKS]
        lines.append(f"{ds if n == (3 if ds == 'CipherSpectrum' else 1) else ''} & {n} & " + " & ".join(f"{x:.1f}" for x in f) + " & "
                     + " & ".join(f"{x:.1f}" for x in s) + r" \\")
    if i < len(DS) - 1:
        lines.append(r"\midrule")
write("S6_testmask", "\n".join(lines))

masked = pd.read_csv(f"{AN}/masked.csv")
unmasked = stat_table("masked_unmasked_same_seeds")
unseen_masked = stat_table("masked_unseen_conn")
cost_fme = stat_table("masked_cost_fme")
cost_single = stat_table("masked_cost_single")
lines, cost_lines = [], []
prev = None
for r in masked.itertuples():
    ds, n = r.dataset, int(r.exit)
    seeds = json.loads(r.seeds)
    if prev is not None and ds != prev:
        lines.append(r"\midrule")
        cost_lines.append(r"\midrule")
    label = ds if ds != prev else ""
    prev = ds
    single = ms(scores("Single", r.baseline_variant, ds, n, seeds=seeds)) + (DAG if r.baseline_variant == "maskall_stepmatch" else "")
    lines.append(f"{label} & {n} & {len(seeds)} & {ms(scores('FME', 'maskall', ds, n, seeds=seeds))} & {single} & {margin(r)} & "
                 f"{pval(r.t_p_holm)} & {sg(100 * unmasked[(ds, n)].diff_mean)} & {sg(100 * unseen_masked[(ds, n)].diff_mean)} \\\\")
    cf, cs = cost_fme[(ds, n)], cost_single[(ds, n)]
    cost_lines.append(f"{label} & {n} & {len(seeds)} & {100 * cf.a_mean:.1f} & {100 * cf.b_mean:.1f} & {sg(-100 * cf.diff_mean)} & "
                      f"{100 * cs.a_mean:.1f} & {100 * cs.b_mean:.1f} & {sg(-100 * cs.diff_mean)} \\\\")
write("S6_retrain", "\n".join(lines))
write("S6_cost", "\n".join(cost_lines))

# ---------------------------------------------------------------- S7: step-matched budget, per-exit checkpoints, eligibility, long budget
lines = []
for i, ds in enumerate(DS[:3]):
    for n in [2, 3, 5]:
        o = ps[(ps.model == "Single") & (ps.variant == "orig") & (ps.dataset == ds) & (ps.exit == n) & (ps.eval_set == "test") & ps.seed.isin(SEEDS)]
        s = ps[(ps.model == "Single") & (ps.variant == "stepmatch") & (ps.dataset == ds) & (ps.exit == n) & (ps.eval_set == "test") & ps.seed.isin(SEEDS)]
        all_flows = int(ps[(ps.model == "Single") & (ps.variant == "orig") & (ps.dataset == ds) & (ps.exit == 1)].train_flows.iloc[0])
        rho = s.train_flows.iloc[0] / all_flows
        e_n, p_n = min(int(np.ceil(25 / rho)), 100), min(int(np.ceil(6 / rho)), 24)
        lines.append(f"{ds if n == 2 else ''} & {n} & {100 * rho:.1f} & {e_n} & {p_n} & {o.optimizer_steps.mean():,.0f} & {s.optimizer_steps.mean():,.0f} & "
                     f"{o.best_epoch.mean():.1f} & {s.best_epoch.mean():.1f} & {o.macro_f1.mean():.3f} & {s.macro_f1.mean():.3f} \\\\")
    if i < 2:
        lines.append(r"\midrule")
write("S7_stepmatch", "\n".join(lines))

ckpt = pd.read_csv(f"{AN}/per_exit_checkpoint.csv")
order = {d: i for i, d in enumerate(DS)}
ckpt = ckpt.assign(o=ckpt.dataset.map(order)).sort_values(["o", "exit"])
lines = []
prev = None
for r in ckpt.itertuples():
    if prev is not None and r.dataset != prev:
        lines.append(r"\midrule")
    lines.append(f"{r.dataset if r.dataset != prev else ''} & {r.exit} & {sg(100 * r.mean_diff, 2)} & {100 * r.sd_diff:.2f} & {r.same_epoch} of {r.runs} \\\\")
    prev = r.dataset
write("S7_checkpoint", "\n".join(lines))

elig = pd.read_csv(f"{AN}/eligibility_matched.csv")
elig = elig.assign(o=elig.dataset.map(order)).sort_values(["o", "exit"])
lines = []
prev = None
for r in elig.itertuples():
    assert r.seeds == 5, (r.dataset, r.exit)
    if prev is not None and r.dataset != prev:
        lines.append(r"\midrule")
    cells = []
    for name in ["objective", "shorter_flows"]:
        x = SimpleNamespace(diff_mean=getattr(r, f"{name}_diff_mean"), clear=getattr(r, f"{name}_clear"),
                            t_p_holm=getattr(r, f"{name}_t_p_holm"))
        cells += [margin(x), pval(x.t_p_holm)]
    lines.append(f"{r.dataset if r.dataset != prev else ''} & {r.exit} & {r.fme_orig:.3f} & {r.elig_mean_rule:.3f} & {r.elig_exit_rule:.3f} & "
                 f"{r.single_stepmatch:.3f} & " + " & ".join(cells) + r" \\")
    prev = r.dataset
write("S7_eligibility", "\n".join(lines))

long50 = stat_table("long50")
long50_fme = stat_table("long50_vs_orig_fme")
long50_single = stat_table("long50_vs_orig_single")
lines = []
for i, ds in enumerate(["ISCXVPN2016", "CSTNET-TLS1.3"]):
    for n in EXITS:
        r = long50[(ds, n)]
        lines.append(f"{ds if n == 1 else ''} & {n} & {int(r.n_seeds)} & {ms(scores('FME', 'long50', ds, n, seeds=SEEDS5))} & "
                     f"{ms(scores('Single', 'long50', ds, n, seeds=SEEDS5))} & {margin(r)} & {pval(r.t_p_holm)} & "
                     f"{sg(100 * long50_fme[(ds, n)].diff_mean)} & {sg(100 * long50_single[(ds, n)].diff_mean)} \\\\")
    if i == 0:
        lines.append(r"\midrule")
write("S7_long50", "\n".join(lines))

# ---------------------------------------------------------------- S4: screened variants, validation and test
variants = pd.read_csv(f"{RESULTS_IN}/exit_variant_scores.csv")
cols = [("USTC-TFC2016", 1), ("USTC-TFC2016", 2), ("USTC-TFC2016", 3), ("USTC-TFC2016", 5), ("CipherSpectrum", 3), ("CipherSpectrum", 5)]
models = ["FlowMamba-Early, as selected", "Exit-specific projections", "Learned exit weights", "Both changes",
          "Two shared and two exit-specific layers"]
lines = []
for block, col in [("Validation", "val_macro_f1"), ("Test", "test_macro_f1")]:
    lines.append(rf"\multicolumn{{7}}{{l}}{{{block} macro-F1}} \\")
    for m in models:
        vals = []
        for ds, n in cols:
            v = variants[(variants.model == m) & (variants.dataset == ds) & (variants.exit == n)][col]
            assert len(v) == 1
            vals.append(f"{float(v.iloc[0]):.3f}")
        lines.append(f"{m} & " + " & ".join(vals) + r" \\")
    if block == "Validation":
        lines.append(r"\midrule")
write("S4_variants", "\n".join(lines))
