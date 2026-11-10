"""Figures of the paper, drawn only from saved result files.

Fig1     model structure and the training rule of the exits (schematic)
Fig2     serving four exits with one model or with four (schematic)
Fig3     share of training flows by number of real packets
Fig4     macro-F1 per exit, FME against the conservative single-exit baseline, ten seeds
Fig5     margins with seed intervals and connection-clustered bootstrap intervals, ten seeds
Fig6     per-class F1 differences at exits 3 and 5, ten seeds
Fig7     pooled confusion matrices at exit 5, ten seeds
Fig8     deployment cost on CSTNET-TLS1.3
S5_FigA  training trajectories over 150 epochs (S5 Appendix)

Each figure is designed 5.4 in wide and exported as a 600 dpi TIFF scaled up to at most 7.5 in
wide, with the scale also capped so that the largest type stays at 12 pt. save() checks every
text object against the 8 to 12 pt range and the 8.75 in height limit at the exported size, and
the checks are written to figures_tiff/FIGURE_CHECK.md.

Reads RESULTS_IN: per_seed.csv, analysis/, dataset_stats.json, deployment_benchmark.json and
long_training/. Writes OUTPUT_ROOT/figures/<name>.pdf and OUTPUT_ROOT/figures_tiff/<name>.tif.

    python src/make_figures.py            # all figures
    python src/make_figures.py 4 5        # Fig4 and Fig5 only
"""
import ast
import csv
import io
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

from paths import OUTPUT_ROOT, RESULTS_IN

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

AN = f"{RESULTS_IN}/analysis"
STATS = f"{RESULTS_IN}/dataset_stats.json"
DEPLOY = f"{RESULTS_IN}/deployment_benchmark.json"
TRAJ = f"{RESULTS_IN}/long_training/{{ds}}_seed20260921.csv"
OUT_PDF = f"{OUTPUT_ROOT}/figures"
OUT_TIF = f"{OUTPUT_ROOT}/figures_tiff"
WIDTH = 5.4            # design width in inches
FINAL = 7.5            # exported width in inches
DPI = 600
MIN_PT = 8.0
MAX_PT = 12.0

BLUE, BLUE_LIGHT = "#0F4D92", "#E3ECF7"
RED = "#C2452D"
GREY_LIGHT, GREY_MID, GREY_DARK, BLACK = "#CFCECE", "#767676", "#4D4D4D", "#272727"
RAMP5 = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
RAMP4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
DATASETS = ["iscxvpn", "ciciot2022", "ustctfc2016", "cipherspectrum", "cstnet-tls1.3"]
DS_NAME = {"iscxvpn": "ISCXVPN2016", "ciciot2022": "CICIoT2022", "ustctfc2016": "USTC-TFC2016",
           "cipherspectrum": "CipherSpectrum", "cstnet-tls1.3": "CSTNET-TLS1.3"}
DATASET_STYLE = {"iscxvpn": ("#B64342", "o"), "ciciot2022": ("#42949E", "s"), "ustctfc2016": ("#9A4D8E", "^"),
                 "cipherspectrum": ("#E28E2C", "D"), "cstnet-tls1.3": ("#4D4D4D", "v")}
EXITS = [1, 2, 3, 5]
LOW = {("cipherspectrum", 1), ("cipherspectrum", 2)}
SEEDS = list(range(20260921, 20260931))
os.makedirs(OUT_PDF, exist_ok=True)
os.makedirs(OUT_TIF, exist_ok=True)
REPORT = []


def save(fig, name, pad=0.03):
    """Write the vector PDF and a 7.5 in, 600 dpi, LZW-compressed RGB TIFF; check the type size."""
    fig.savefig(f"{OUT_PDF}/{name}.pdf", bbox_inches="tight", pad_inches=pad)
    renderer = fig.canvas.get_renderer()
    bbox = fig.get_tightbbox(renderer)
    width_in = bbox.width + 2 * pad
    sizes = [t.get_fontsize() for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text().strip()]
    scale = min(FINAL / width_in, MAX_PT / max(sizes))
    smallest, largest = min(sizes) * scale, max(sizes) * scale
    target = round(width_in * scale * DPI)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI * scale, bbox_inches="tight", pad_inches=pad, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    if img.width != target:
        img = img.resize((target, round(img.height * target / img.width)), Image.LANCZOS)
    img.save(f"{OUT_TIF}/{name}.tif", compression="tiff_lzw", dpi=(DPI, DPI))
    img.resize((img.width // 4, img.height // 4), Image.LANCZOS).save(f"{OUT_TIF}/preview_{name}.png")
    w_in, h_in = img.width / DPI, img.height / DPI
    ok = smallest >= MIN_PT - 0.05 and largest <= MAX_PT + 0.05 and h_in <= 8.75 and 2.63 <= w_in <= 7.5
    REPORT.append((name, img.width, img.height, round(w_in, 2), round(h_in, 2), round(smallest, 1), round(largest, 1), ok))
    print(f"{name}: {img.width}x{img.height} px, {w_in:.2f} x {h_in:.2f} in, type {smallest:.1f}-{largest:.1f} pt, {'ok' if ok else 'CHECK'}")


def panel_label(ax, text, x=-0.12, y=1.04):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def box(ax, x, y, w, h, text, fc, ec, fs=6, color=BLACK, weight="normal", lw=0.8, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, fontweight=weight,
            linespacing=1.15)


def arrow(ax, p, q, color=GREY_DARK, lw=0.8, style="-|>", ms=6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, lw=lw, color=color, shrinkA=0, shrinkB=0))


# ---------------------------------------------------------------- results of the 10-seed analysis
def load_results():
    ps = pd.read_csv(f"{RESULTS_IN}/per_seed.csv")
    ps = ps[(ps.status == "ok") & (ps.eval_set == "test") & ps.seed.isin(SEEDS)]
    main = pd.read_csv(f"{AN}/main_vs_conservative.csv")
    base = {(r.dataset, int(r.exit)): r.baseline_variant for r in main.itertuples()}
    return ps, main, base


def seed_scores(ps, model, variant, ds, n):
    s = ps[(ps.model == model) & (ps.variant == variant) & (ps.dataset == DS_NAME[ds]) & (ps.exit == n)]
    s = s.set_index("seed").macro_f1
    assert len(s) == len(SEEDS), (model, variant, ds, n, len(s))
    return np.array([s[x] for x in SEEDS])


# ---------------------------------------------------------------- Fig 1: architecture and masking
def fig_architecture():
    fig, ax = plt.subplots(figsize=(WIDTH, 3.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 74)
    ax.axis("off")
    ax.text(0.5, 72, "A", fontsize=8, fontweight="bold", va="center")
    ax.text(4, 72, "Structure of the shared model", fontsize=7, va="center")
    y0, h = 47, 15
    for k in range(5):
        box(ax, 1 + k * 3.7, y0 + 1.5, 3.2, 12, "", "#F2F2F2", GREY_MID, lw=0.6)
        ax.text(1 + k * 3.7 + 1.6, y0 + 7.5, f"P{k + 1}", fontsize=6, ha="center", va="center", color=GREY_DARK)
    ax.text(10.3, y0 - 2.2, "5 packets\n320 bytes each", fontsize=6, ha="center", va="top", color=GREY_DARK, linespacing=1.15)
    arrow(ax, (20.4, y0 + 7.5), (23, y0 + 7.5))
    box(ax, 23, y0 + 1.5, 10.5, 12, "Byte\nembedding\n256$\\rightarrow$256", "white", GREY_MID, fs=6)
    arrow(ax, (33.9, y0 + 7.5), (36.6, y0 + 7.5))
    ax.add_patch(FancyBboxPatch((36.6, y0 - 1), 26, h + 2, boxstyle="round,pad=0,rounding_size=0.8", fc=BLUE_LIGHT, ec=BLUE, lw=0.8))
    ax.text(49.6, y0 + h + 2.8, r"$\times$ 4 residual blocks", fontsize=6.2, ha="center", va="bottom", color=BLUE, fontweight="bold")
    box(ax, 38.4, y0 + 3, 9, 9, "Layer\nnorm", "white", GREY_MID, fs=6)
    arrow(ax, (47.6, y0 + 7.5), (50.2, y0 + 7.5))
    box(ax, 50.2, y0 + 3, 10.6, 9, "Mamba-2\nlayer", "white", BLUE, fs=6)
    ax.add_patch(FancyArrowPatch((37.6, y0 + 12.4), (61.7, y0 + 12.4), arrowstyle="-", lw=0.6, color=GREY_MID,
                                 connectionstyle="arc3,rad=0", shrinkA=0, shrinkB=0))
    ax.text(49.6, y0 + 13.4, "residual sum", fontsize=6, ha="center", va="bottom", color=GREY_DARK)
    arrow(ax, (63.0, y0 + 7.5), (65.6, y0 + 7.5))
    box(ax, 65.6, y0 + 1.5, 9.6, 12, "Final\nlayer\nnorm", "white", GREY_MID, fs=6)
    arrow(ax, (75.6, y0 + 7.5), (78.4, y0 + 7.5))
    ax.text(77.2, y0 - 3.4, "hidden state at\nbyte $320N$", fontsize=6, ha="center", va="top", color=GREY_DARK, linespacing=1.15)
    box(ax, 78.4, y0 - 1.2, 9.6, 17.4, "Shared\nlinear\n+ GELU", "white", GREY_MID, fs=6)
    for i, n in enumerate([1, 2, 3, 5]):
        yy = y0 + 17 - 6.8 * i
        arrow(ax, (88.2, y0 + 7.5), (91.0, yy), lw=0.6)
        box(ax, 91.0, yy - 2.4, 8.4, 4.8, f"Head {n}", "white", BLUE, fs=6)

    ax.text(0.5, 36.5, "B", fontsize=8, fontweight="bold", va="center")
    ax.text(4, 36.5, "Which flows train which exit", fontsize=7, va="center")
    gx, gy, cw, rh = 19, 3, 8.5, 5.2
    ax.text(gx + 2 * cw, gy + 5 * rh + 4.2, "Exit $N$", fontsize=6.2, ha="center", va="center", color=GREY_DARK)
    for j, n in enumerate([1, 2, 3, 5]):
        ax.text(gx + j * cw + cw / 2, gy + 5 * rh + 1.2, f"{n}", fontsize=6.3, ha="center", va="center", fontweight="bold")
    ax.text(gx - 3, gy + 5 * rh + 1.2, "real packets $n$", fontsize=6.2, ha="right", va="center", color=GREY_DARK)
    for i, npk in enumerate([1, 2, 3, 4, 5]):
        yy = gy + (4 - i) * rh
        ax.text(gx - 3, yy + rh / 2, f"{npk}", fontsize=6.3, ha="right", va="center", fontweight="bold")
        for j, n in enumerate([1, 2, 3, 5]):
            elig = npk >= n
            ax.add_patch(Rectangle((gx + j * cw, yy), cw, rh, fc=BLUE_LIGHT if elig else "white", ec=GREY_LIGHT, lw=0.5))
            if elig:
                ax.scatter([gx + j * cw + cw / 2], [yy + rh / 2], s=9, color=BLUE, zorder=3)
            else:
                ax.text(gx + j * cw + cw / 2, yy + rh / 2, "$-$", fontsize=6.2, ha="center", va="center", color=GREY_MID)
    ax.text(57, gy + 4.4 * rh, "A flow with $n$ real packets adds cross-entropy\nonly at the exits with $N \\leq n$. Padding beyond\nits true length is never used as evidence. The\nbatch loss averages the flows within each exit,\nthen averages over the exits that have at least\none flow in the batch.",
            fontsize=6.2, ha="left", va="top", color=BLACK, linespacing=1.3)
    ax.text(57, gy + 0.1 * rh, "A single-exit model for $N$ trains only on\nthe rows with $n \\geq N$, in the column of its\nown exit.", fontsize=6.2,
            ha="left", va="bottom", color=GREY_DARK, linespacing=1.3)
    save(fig, "Fig1")


# ---------------------------------------------------------------- Fig 2: serving four exits
def fig_overview():
    fig, ax = plt.subplots(figsize=(WIDTH, 4.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 80)
    ax.axis("off")
    ax.text(0.5, 78, "A", fontsize=8, fontweight="bold", va="center")
    ax.text(4, 78, "One shared model (FlowMamba-Early)", fontsize=7, va="center")
    xs = [6 + 17.5 * k for k in range(5)]
    w = 13
    for k, x in enumerate(xs):
        box(ax, x, 42, w, 5, f"Packet {k + 1}", "#F2F2F2", GREY_MID, fs=6.5)
        box(ax, x, 51, w, 6, "Mamba-2", BLUE_LIGHT, BLUE, fs=6.5)
        arrow(ax, (x + w / 2, 47), (x + w / 2, 51))
        if k < 4:
            arrow(ax, (x + w, 54), (xs[k + 1], 54), color=BLUE, lw=1.0)
        if k + 1 in EXITS:
            box(ax, x, 61.5, w, 4.5, f"Head {k + 1}", "white", BLUE, fs=6.5)
            arrow(ax, (x + w / 2, 57), (x + w / 2, 61.5))
            ax.text(x + w / 2, 71, rf"$\hat{{y}}$ after {k + 1}", ha="center", va="center", fontsize=6.5)
            arrow(ax, (x + w / 2, 66), (x + w / 2, 69.3))
    ax.text((xs[3] + w + xs[4]) / 2, 57.8, "state", fontsize=6.2, color=BLUE, ha="center", va="bottom")
    ax.text(47.5, 38.5, "5 packet updates per flow, one set of weights", ha="center", va="center",
            fontsize=6.5, color=BLUE, fontweight="bold")
    ax.plot([0, 100], [35.5, 35.5], color=GREY_LIGHT, lw=0.6)
    ax.text(0.5, 32.5, "B", fontsize=8, fontweight="bold", va="center")
    ax.text(4, 32.5, "Four single-exit models (FlowMamba-Single-N)", fontsize=7, va="center")
    pw, gap = 7.5, 1.2
    for row, n in enumerate([1, 2, 3, 5]):
        y = 24.5 - row * 6.5
        ax.text(6, y + 2.2, f"N = {n}", fontsize=6.5, va="center", ha="left")
        for k in range(n):
            box(ax, 16 + k * (pw + gap), y, pw, 4.4, f"P{k + 1}", "#F2F2F2", GREY_MID, fs=6.2)
        x_end = 16 + n * (pw + gap) - gap
        box(ax, 64, y - 0.2, 14, 4.8, f"Model {n}", "#EFEFEF", GREY_DARK, fs=6.5)
        arrow(ax, (x_end + 0.5, y + 2.2), (64, y + 2.2))
        ax.text(84, y + 2.2, rf"$\hat{{y}}_{n}$", ha="center", va="center", fontsize=6.5)
        arrow(ax, (78, y + 2.2), (81.8, y + 2.2))
    ax.text(47.5, 0.8, "1 + 2 + 3 + 5 = 11 packet updates per flow, four sets of weights", ha="center",
            va="center", fontsize=6.5, color=GREY_DARK, fontweight="bold")
    save(fig, "Fig2")


# ---------------------------------------------------------------- Fig 3: packet-count distribution
def fig_packets():
    st = json.load(open(STATS))
    fig, ax = plt.subplots(figsize=(4.75, 2.35))
    order = DATASETS[::-1]
    for r, ds in enumerate(order):
        tr = st[ds]["splits"]["train"]
        tot = tr["n"]
        left = 0.0
        for k in range(1, 6):
            share = 100.0 * tr["num_packet_capped"].get(str(k), 0) / tot
            if share > 0:
                ax.barh(r, share, left=left, height=0.62, color=RAMP5[k - 1], edgecolor="white", linewidth=0.8)
                if share >= 4.0:
                    ax.text(left + share / 2, r, f"{share:.0f}" if share >= 9.5 else f"{share:.1f}", ha="center", va="center", fontsize=6.2,
                            color=BLACK if k <= 2 else "white")
            left += share
        ax.text(101.5, r, f"{tot:,} flows, {len(st[ds]['classes'])} classes", ha="left", va="center", fontsize=6.2, color=GREY_DARK)
    ax.set_yticks(range(5))
    ax.set_yticklabels([DS_NAME[d] for d in order])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of training flows (%)")
    ax.tick_params(axis="y", length=0)
    handles = [mpl.patches.Patch(color=RAMP5[k - 1], label=f"{k}" if k < 5 else "5 or more") for k in range(1, 6)]
    ax.legend(handles=handles, title="Real packets in the flow", ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              columnspacing=1.0, handlelength=1.0, handletextpad=0.4, title_fontsize=6.5, borderaxespad=0.2)
    save(fig, "Fig3")


# ---------------------------------------------------------------- Fig 4: accuracy, ten seeds
def fig_accuracy():
    ps, main, base = load_results()
    stat = {(r.dataset, int(r.exit)): r for r in main.itertuples()}
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 5.75))
    fig.subplots_adjust(wspace=0.32, hspace=0.78)
    for i, ds in enumerate(DATASETS):
        ax = axes.flat[i]
        name = DS_NAME[ds]
        fme = {n: seed_scores(ps, "FME", "orig", ds, n) for n in EXITS}
        single = {n: seed_scores(ps, "Single", base.get((name, n), "orig"), ds, n) for n in EXITS}
        values = []
        for scores, color, ls, marker, dx in [(fme, BLUE, "-", "o", -0.08), (single, GREY_MID, "--", "s", 0.08)]:
            m = np.array([scores[n].mean() for n in EXITS])
            s = np.array([scores[n].std(ddof=1) for n in EXITS])
            ax.fill_between(EXITS, m - s, m + s, color=color, alpha=0.15, lw=0)
            ax.plot(EXITS, m, ls=ls, color=color, marker=marker, ms=3.2, lw=1.1, zorder=4)
            for n in EXITS:
                ax.scatter([n + dx] * len(SEEDS), scores[n], s=4, facecolors="none", edgecolors=color, linewidths=0.45, zorder=3)
                values += list(scores[n])
            values += list(m - s) + list(m + s)
        lo, hi = min(values), max(values)
        span = hi - lo
        ax.set_ylim(max(0.0, lo - 0.08 * span), hi + 0.32 * span)
        ax.set_xticks(EXITS)
        ax.set_xlim(0.6, 5.4)
        ax.set_xlabel("Exit (packets)")
        if i % 2 == 0:
            ax.set_ylabel("Macro-F1")
        ax.set_title(name, loc="left", pad=3)
        panel_label(ax, "ABCDE"[i], x=-0.18)
        for n in EXITS:
            if (ds, n) in LOW:
                continue
            r = stat[(name, n)]
            mark = "*" if r.t_p_holm < 0.05 else ""
            ax.text(n, 0.93, f"{100 * r.diff_mean:+.1f}{mark}", transform=ax.get_xaxis_transform(), ha="center", va="center",
                    fontsize=6.2, fontweight="bold" if r.clear else "normal", color=BLACK if r.clear else GREY_MID)
        if ds == "cipherspectrum":
            ax.axvspan(0.6, 2.5, color=GREY_LIGHT, alpha=0.35, lw=0)
            ax.text(1.55, 0.45, "near\nchance", transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=6.2, color=GREY_DARK)
    handles = [Line2D([], [], color=BLUE, marker="o", ms=3.2, lw=1.1, label="FlowMamba-Early (one model)"),
               Line2D([], [], color=GREY_MID, marker="s", ms=3.2, lw=1.1, ls="--", label="FlowMamba-Single-N, conservative baseline")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995))

    ax = axes.flat[5]
    n_train = {}
    for ds in DATASETS:
        for n in EXITS:
            t = ps[(ps.model == "Single") & (ps.variant == "orig") & (ps.dataset == DS_NAME[ds]) & (ps.exit == n)].train_flows
            n_train[(ds, n)] = int(t.iloc[0])
    for j, ds in enumerate(DATASETS):
        color, marker = DATASET_STYLE[ds]
        for n in EXITS:
            if (ds, n) in LOW:
                continue
            r = stat[(DS_NAME[ds], n)]
            frac = 100 * n_train[(ds, n)] / n_train[(ds, 1)]
            if frac > 99:
                frac = 100 + 2.2 * (j - 2)
            ax.scatter(frac, 100 * r.diff_mean, marker=marker, s=17, facecolors=color if r.clear else "white",
                       edgecolors=color, linewidths=0.8, zorder=3)
        ax.scatter([], [], marker=marker, s=17, facecolors=color, edgecolors=color, label=DS_NAME[ds])
    ax.axhline(0, color=GREY_MID, lw=0.6, ls=":")
    ax.set_xlim(18, 110)
    ax.set_xlabel("Training flows reaching\nthis exit (%)")
    ax.set_ylabel("Margin (points)")
    ax.set_title("Margin and training data", loc="left", pad=3)
    ax.set_ylim(-2.5, 19.5)
    ax.legend(loc="upper right", fontsize=6, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.2)
    panel_label(ax, "F", x=-0.18)
    save(fig, "Fig4")


# ---------------------------------------------------------------- Fig 5: margins and intervals, ten seeds
def fig_margins():
    _, main, _ = load_results()
    order = {DS_NAME[d]: i for i, d in enumerate(DATASETS)}
    rows = sorted(main.itertuples(), key=lambda r: (order[r.dataset], r.exit))
    fig, ax = plt.subplots(figsize=(WIDTH, 4.4))
    y, prev = 0, None
    ticks, labels, seps = [], [], []
    for r in rows:
        if prev is not None and r.dataset != prev:
            y += 0.7
            seps.append(y - 0.65)
        prev = r.dataset
        m, sd = 100 * r.diff_mean, 100 * max(r.a_sd, r.b_sd)
        lo, hi = ast.literal_eval(r.cluster_boot_ci)
        sig = r.t_p_holm < 0.05
        ax.add_patch(Rectangle((0, y - 0.34), sd, 0.68, fc="#EEEEEE", ec="none", zorder=1))
        ax.plot([100 * r.ci_low, 100 * r.ci_high], [y - 0.12, y - 0.12], color=BLUE if sig else GREY_MID, lw=1.6, zorder=2,
                solid_capstyle="butt")
        ax.plot([100 * lo, 100 * hi], [y + 0.2, y + 0.2], color=BLACK, lw=0.7, zorder=2, solid_capstyle="butt")
        ax.scatter([m], [y - 0.12], s=18, zorder=3, marker="o", facecolors=BLUE if sig else "white",
                   edgecolors=BLUE if sig else GREY_DARK, linewidths=0.9)
        ticks.append(y)
        labels.append(f"{r.dataset}, $N={r.exit}$")
        y += 1
    ax.axvline(0, color=GREY_MID, lw=0.7)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6.2)
    ax.invert_yaxis()
    ax.set_ylim(y, -0.9)
    ax.set_xlabel("Macro-F1 margin, FME $-$ conservative single-exit baseline (points)")
    ax.tick_params(axis="y", length=0)
    for s in seps:
        ax.axhline(s, color=GREY_LIGHT, lw=0.5)
    handles = [Line2D([], [], marker="o", ls="-", color=BLUE, mfc=BLUE, ms=3.8, lw=1.6, label="seed interval, Holm $p<0.05$"),
               Line2D([], [], marker="o", ls="-", color=GREY_MID, mfc="white", mec=GREY_DARK, ms=3.8, lw=1.6, label="seed interval, not significant"),
               Line2D([], [], ls="-", color=BLACK, lw=0.7, label="connection bootstrap interval"),
               Rectangle((0, 0), 1, 1, fc="#EEEEEE", ec="none", label="zero to larger seed SD")]
    ax.legend(handles=handles, loc="lower right", fontsize=6.2, handlelength=1.8, borderaxespad=0.2)
    save(fig, "Fig5")


# ---------------------------------------------------------------- Fig 6: per-class F1 differences, ten seeds
def fig_perclass():
    CAP = 30
    pc = json.load(open(f"{AN}/per_class.json"))
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    rng = np.random.default_rng(7)
    for ax, n, letter in [(axes[0], 3, "A"), (axes[1], 5, "B")]:
        ax.axhline(0, color=GREY_MID, lw=0.7)
        for j, ds in enumerate(DATASETS):
            rec = pc[f"{DS_NAME[ds]}|{n}"]
            d = np.array(rec["diff_points"], float)
            xj = j + rng.uniform(-0.26, 0.26, len(d))
            pos = (d >= 0) & (d <= CAP)
            ax.scatter(xj[pos], d[pos], s=7, marker="o", facecolors=BLUE, edgecolors="none", alpha=0.85, zorder=3)
            for xv, v in zip(xj[d > CAP], d[d > CAP]):   # drawn at the top edge with the value written next to it
                ax.scatter([xv], [CAP], s=14, marker="^", facecolors=BLUE, edgecolors="none", zorder=3)
                ax.text(xv + 0.12, CAP, f"+{v:.0f}", fontsize=6, color=BLUE, va="center", ha="left")
            ax.scatter(xj[~pos], d[~pos], s=8, marker="D", facecolors="white", edgecolors=RED, linewidths=0.6, zorder=3)
            ax.plot([j - 0.34, j + 0.34], [np.median(d)] * 2, color=BLACK, lw=1.1, zorder=4)
            up, down = int((d > 0.5).sum()), int((d < -0.5).sum())
            assert (up, down) == (rec["up"], rec["down"])
            ax.text(j, 1.03, f"{up} up\n{down} down", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6,
                    color=GREY_DARK, linespacing=1.1)
        ax.set_xticks(range(5))
        ax.set_xticklabels(["ISCXVPN\n2016", "CICIoT\n2022", "USTC-\nTFC2016", "Cipher-\nSpectrum", "CSTNET-\nTLS1.3"], fontsize=6)
        ax.set_xlim(-0.6, 4.6)
        ax.set_title(f"Exit $N={n}$", loc="left", pad=24, fontsize=7)
        panel_label(ax, letter, x=-0.02 if n == 3 else -0.04, y=1.22)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylim(top=CAP + 3)
    axes[0].set_ylabel("F1 difference, FME $-$ baseline (points)")
    axes[1].tick_params(axis="y", length=0)
    axes[1].spines["left"].set_visible(False)
    handles = [Line2D([], [], marker="o", ls="", mfc=BLUE, mec="none", ms=3.6, label="class gains"),
               Line2D([], [], marker="D", ls="", mfc="white", mec=RED, mew=0.6, ms=3.6, label="class loses"),
               Line2D([], [], color=BLACK, lw=1.1, label="median over classes")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05), handletextpad=0.4, columnspacing=1.6)
    save(fig, "Fig6")


# ---------------------------------------------------------------- Fig 7: confusion matrices, ten seeds
SEQ = mpl.colors.LinearSegmentedColormap.from_list("blue_seq", ["#f7fafe", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])


def conf_panel(ax, cm_raw, labels, title, fs_cell=6, fs_tick=6, rot=90):
    rown = cm_raw / np.maximum(cm_raw.sum(1, keepdims=True), 1)
    ax.imshow(rown, cmap=SEQ, vmin=0, vmax=1, aspect="equal")
    k = len(labels)
    for i in range(k):
        for j in range(k):
            v = rown[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=fs_cell, color="white" if v > 0.55 else BLACK)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(labels, rotation=rot, fontsize=fs_tick)
    ax.set_yticklabels(labels, fontsize=fs_tick)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=6.8, pad=3)
    ax.set_xlabel("Predicted class", fontsize=6.3)
    ax.set_ylabel("True class", fontsize=6.3)


def fig_confusion():
    cf = json.load(open(f"{AN}/confusion.json"))
    st = json.load(open(STATS))
    names = {ds: [st[ds]["classes"][str(i)] for i in range(len(st[ds]["classes"]))] for ds in ["iscxvpn", "ciciot2022"]}
    short = {"iscxvpn": [n.replace("VPN-", "") for n in names["iscxvpn"]],
             "ciciot2022": ["Flood", "RTSP BF", "Audio", "Power other", "Cameras", "Home autom."]}
    assert names["ciciot2022"][1].endswith("RTSP Brute Force") and names["ciciot2022"][3].endswith("Power-Other")
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.3))
    fig.subplots_adjust(wspace=0.5, hspace=0.68)
    for r, (ds, letters) in enumerate([("iscxvpn", "AB"), ("ciciot2022", "CD")]):
        for c, (fam, nm) in enumerate([("fme", "FlowMamba-Early"), ("baseline", "Single-5")]):
            ax = axes[r, c]
            rec = cf[f"{DS_NAME[ds]}|{fam}"]
            assert fam == "fme" or rec["variant"] == "orig"
            conf_panel(ax, np.array(rec["counts"], float), short[ds], f"{DS_NAME[ds]}, {nm}")
            panel_label(ax, letters[c], x=-0.42, y=1.1)
    cax = fig.add_axes([0.93, 0.35, 0.015, 0.3])
    sm = mpl.cm.ScalarMappable(cmap=SEQ, norm=mpl.colors.Normalize(0, 100))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Row share (%)", fontsize=6.3)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_visible(False)
    save(fig, "Fig7")


# ---------------------------------------------------------------- Fig 8: deployment cost
def fig_deployment():
    d = json.load(open(DEPLOY))
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.6))
    fig.subplots_adjust(wspace=0.55, hspace=0.68)
    ax = axes[0, 0]
    naive = d["bs128"]["naive_exit_ms"]
    update = d["bs128"]["incremental_packet_update_ms"]
    ax.plot(EXITS, [naive[str(n)] for n in EXITS], color=GREY_MID, marker="s", ms=3.2, lw=1.1, ls="--")
    ax.plot(range(1, 6), [update[str(k)] for k in range(1, 6)], color=BLUE, marker="o", ms=3.2, lw=1.1)
    ax.text(5.1, naive["5"], "recompute\nprefix", fontsize=6.2, color=GREY_DARK, va="center", ha="left")
    ax.text(5.1, update["5"] + 3, "incremental\nupdate", fontsize=6.2, color=BLUE, va="center", ha="left")
    ax.set_xticks(range(1, 6))
    ax.set_xlim(0.7, 6.6)
    ax.set_ylim(0, max(naive.values()) * 1.15)
    ax.set_xlabel("Packets received")
    ax.set_ylabel("Time per step (ms)")
    ax.set_title("One decision step, batch 128", loc="left", pad=3)
    panel_label(ax, "A", x=-0.2)
    labels = ["Recompute\nat every exit", "FME\nincremental", "Four single-\nexit\nincremental"]
    keys = ["naive_recompute", "shared_incremental", "specialists_incremental"]
    colors = [GREY_LIGHT, BLUE, GREY_MID]
    for ax, bs, letter in [(axes[0, 1], "bs128", "B"), (axes[1, 0], "bs1", "C")]:
        c = d[bs]["cumulative_all_exits_ms"]
        vals = [c[k] for k in keys]
        bars = ax.bar(range(3), vals, color=colors, width=0.62, edgecolor="none")
        for k, (bar, v) in enumerate(zip(bars, vals)):
            ratio = v / c["shared_incremental"]
            text = f"{v:.1f} ms" if k == 1 else f"{v:.1f} ms\n({ratio:.1f}× FME)"
            ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.03, text, ha="center", va="bottom", fontsize=6.2)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, fontsize=6.2)
        ax.set_ylim(0, max(vals) * 1.4)
        ax.set_ylabel("Time for four exits (ms)")
        ax.set_title(f"Cumulative cost, batch {bs[2:]}", loc="left", pad=3)
        panel_label(ax, letter, x=-0.2)
    ax = axes[1, 1]
    p = d["params"]
    names = ["FME (one model)", "4 × Single-N", "4 × MNB", "4 × NetMamba-N"]
    vals = [p["flowmamba_early"], p["flowmamba_single_n_x4"], p["mambanetburst_n_x4"], p["netmamba_n_x4"]]
    bars = ax.barh(range(4), [v / 1e6 for v in vals], color=[BLUE, GREY_MID, GREY_LIGHT, GREY_LIGHT], height=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.12, bar.get_y() + bar.get_height() / 2, f"{v / 1e6:.2f}M", va="center", fontsize=6.2)
    ax.set_yticks(range(4))
    ax.set_yticklabels(names, fontsize=6.2)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) / 1e6 * 1.3)
    ax.set_xlabel("Parameters for four exits\n(millions)")
    ax.set_title("Model storage", loc="left", pad=3)
    panel_label(ax, "D", x=-0.62)
    save(fig, "Fig8")


# ---------------------------------------------------------------- S5 Appendix Fig A: 150-epoch trajectories
def fig_trajectories():
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.1), sharex=True)
    fig.subplots_adjust(hspace=0.18, wspace=0.3)
    for c, (ds, title) in enumerate([("cipherspectrum", "CipherSpectrum"), ("ustctfc2016", "USTC-TFC2016")]):
        rows = list(csv.DictReader(open(TRAJ.format(ds=ds))))
        ep = np.array([int(r["epoch"]) for r in rows])
        for k, n in enumerate([1, 2, 3, 5]):
            ce = np.array([float(r[f"train_ce_{n}"]) for r in rows])
            vf = np.array([float(r[f"val_f1_{n}"]) for r in rows])
            ev = ep % 5 == 0
            ev[-1] = True
            axes[0, c].plot(ep, np.maximum(ce, 1e-6), color=RAMP4[k], lw=1.1)
            axes[1, c].plot(ep[ev], vf[ev], color=RAMP4[k], lw=1.1, marker="o", ms=1.8)
            if c == 1:
                axes[1, c].text(151.5, vf[-1], f"$N={n}$", fontsize=6.2, color=GREY_DARK, va="center", ha="left")
        axes[0, c].set_yscale("log")
        axes[0, c].set_ylim(1e-6, 8)
        axes[0, c].set_title(title, loc="left", pad=3)
        axes[1, c].set_xlabel("Epoch")
        for r in range(2):
            axes[r, c].axvline(25, color=GREY_MID, lw=0.7, ls="--")
        axes[0, c].text(26.5, 3.5e-6, "common recipe\nstops by epoch 25", fontsize=6.2, color=GREY_DARK, va="bottom", ha="left")
        panel_label(axes[0, c], "AC"[c], x=-0.22, y=1.1)
        panel_label(axes[1, c], "BD"[c], x=-0.22, y=1.05)
        axes[1, c].set_xlim(0, 150)
    axes[0, 0].set_ylabel("Training cross-entropy")
    axes[1, 0].set_ylabel("Validation macro-F1")
    handles = [Line2D([], [], color=RAMP4[k], lw=1.4, label=f"exit {n}") for k, n in enumerate([1, 2, 3, 5])]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04), columnspacing=1.6, handlelength=1.6)
    save(fig, "S5_FigA")


FIGS = {"1": fig_architecture, "2": fig_overview, "3": fig_packets, "4": fig_accuracy, "5": fig_margins,
        "6": fig_perclass, "7": fig_confusion, "8": fig_deployment, "S5A": fig_trajectories}

if __name__ == "__main__":
    for key in sys.argv[1:] or list(FIGS):
        FIGS[key]()
    with open(f"{OUT_TIF}/FIGURE_CHECK.md", "w") as f:
        f.write("| File | Width px | Height px | Width in | Height in | Smallest type pt | Largest type pt | Within size limits |\n|---|---:|---:|---:|---:|---:|---:|---|\n")
        for r in REPORT:
            f.write(f"| {r[0]}.tif | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {'yes' if r[7] else 'no'} |\n")
