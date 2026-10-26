"""The paper's figures, redrawn for print from bench/data and eval/out (no numbers typed in by hand).

    python paper/figures.py                     # -> paper/fig/*.pdf

Grayscale-safe: every series carries a marker or line style as well as its colour.
"""
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import common  # noqa: E402

OUT_DIR = ROOT / "paper" / "fig"
DATA = ROOT / "bench" / "data" / "LAPTOP-9KPB5IBV-20261026-002239.csv"
EVAL = common.OUT
C = common.COLORS
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})


def read(name):
    with open(EVAL / name) as f:
        return list(csv.DictReader(f))


def style(ax):
    common.style(ax)
    ax.set_facecolor("white")
    ax.tick_params(colors=common.INK)


def save(fig, name):
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def crossover():
    """GPU speed-up (T_cpu / T_gpu) against input elements, one panel per op."""
    runs = defaultdict(list)
    with open(DATA) as f:
        for r in csv.DictReader(f):
            if int(r["rep"]) > 0:  # rep 0 is the warm-up the collector records
                runs[r["op"], int(r["elements"]), r["device"]].append(float(r["seconds"]))
    ops = list(dict.fromkeys(k[0] for k in runs))
    fig, axes = plt.subplots(2, 5, figsize=(7.16, 3.2), sharey=True, gridspec_kw={"hspace": 0.55})
    for ax, op in zip(axes.flat, ops):
        sizes = sorted({e for o, e, d in runs if o == op and (o, e, "cpu") in runs and (o, e, "gpu") in runs})
        speed = [statistics.median(runs[op, e, "cpu"]) / statistics.median(runs[op, e, "gpu"]) for e in sizes]
        ax.axhspan(1e-2, 1, color=C["cpu"], alpha=0.08, lw=0)
        ax.axhline(1, color=common.MUTED, lw=0.8, ls="--")
        ax.plot(sizes, speed, color=C["gpu"], lw=1.5, marker="o", ms=3)
        ax.set(xscale="log", yscale="log", ylim=(0.08, 60), title=op)
        style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$T_{cpu}/T_{gpu}$")
    for ax in axes[1]:
        ax.set_xlabel("input elements")
    save(fig, "crossover")


def accuracy():
    """RQ1: accuracy per evaluation, and per held-out op."""
    summary, by_op = read("rq1.csv"), read("rq1_by_op.csv")
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.5, 3.6), gridspec_kw={"height_ratios": [1, 1.6]})
    evals = list(dict.fromkeys(r["evaluation"] for r in summary))
    for i, kind in enumerate(("forest", "tree")):
        vals = [next((float(r["accuracy"]) for r in summary if r["evaluation"] == e and r["model"] == kind), None) for e in evals]
        ys = [j + (i - 0.5) * 0.38 for j in range(len(evals))]
        a.barh([y for y, v in zip(ys, vals) if v is not None], [v for v in vals if v is not None], height=0.34,
               color=C["gpu"] if kind == "forest" else C["cpu"], hatch=None if kind == "forest" else "////",
               edgecolor="white", lw=0.5, label=f"random {kind}" if kind == "forest" else "decision tree")
    a.set_yticks(range(len(evals)), [e.replace("calibration -> benchmark", "calibration only") for e in evals])
    a.invert_yaxis()
    a.set(xlim=(0, 1), xlabel="share of sizes where the faster device was picked")
    a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    style(a)
    rows = sorted((r for r in by_op if r["model"] == "forest"), key=lambda r: float(r["accuracy"]))
    b.barh([r["op"] for r in rows], [float(r["accuracy"]) for r in rows], color=C["gpu"], height=0.6)
    for y, r in enumerate(rows):
        b.text(float(r["accuracy"]) + 0.01, y, f"{float(r['accuracy']):.0%}", va="center", fontsize=6.5)
    b.set(xlim=(0, 1.08), xlabel="leave-one-op-out accuracy (forest)")
    style(b)
    fig.tight_layout()
    save(fig, "rq1")


def confidence():
    """Selective calibration on AutoHetero's data, and per-op confidence against accuracy."""
    sweep, rows = read("confidence_sweep.csv"), read("confidence.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 2.1))
    for ev, color, marker in (("5-fold", C["cpu"], "o"), ("leave-one-op-out", C["gpu"], "s")):
        p = [r for r in sweep if r["evaluation"] == ev]
        a.plot([float(r["measured_share"]) for r in p], [float(r["accuracy"]) for r in p], color=color, marker=marker, ms=3.5, lw=1.5, label=ev)
    a.set(xlabel="share of sizes measured", ylabel="decision accuracy", xlim=(-0.03, 1.03), ylim=(0.65, 1.01))
    a.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.12), fontsize=6)
    style(a)
    loo = [r for r in rows if r["evaluation"] == "leave-one-op-out"]
    for op in sorted({r["op"] for r in loo}):
        p = [r for r in loo if r["op"] == op]
        acc = sum(r["pick"] == r["best"] for r in p) / len(p)
        conf = statistics.median(float(r["confidence"]) for r in p)
        b.scatter(conf, acc, s=14, color=C["gpu"], edgecolor="white", lw=0.5, zorder=3)
        if op in ("fft", "sort", "filter", "matmul", "groupby"):
            b.annotate(op, (conf, acc), xytext=(3, -2), textcoords="offset points", fontsize=6.5)
    b.set(xlabel="median confidence", ylabel="accuracy (held-out op)", xlim=(0.5, 1), ylim=(0, 1.05))
    style(b)
    fig.tight_layout()
    save(fig, "confidence")


def rq2():
    """AutoHetero against always-GPU per workload and condition (sort 10M left out, CARRYOVER.md row 12)."""
    rows = [r for r in read("rq2.csv") if not (r["workload"] == "sort" and r["size"] == "10000000")]
    conds = ["idle", "gpu", "cpu", "battery"]
    names = {"idle": "idle", "gpu": "GPU\nstress", "cpu": "CPU\nstress", "battery": "battery"}
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    ax.axhline(1, color=common.MUTED, lw=0.8, ls="--")
    for i, c in enumerate(conds):
        v = [float(r["speedup_vs_gpu"]) for r in rows if r["condition"] == c]
        jitter = [(j % 5 - 2) * 0.05 for j in range(len(v))]
        ax.scatter([i + d for d in jitter], v, s=10, color=C["autohetero"], edgecolor="white", lw=0.4, zorder=3)
        g = math.exp(statistics.mean(math.log(x) for x in v))
        ax.plot([i - 0.28, i + 0.28], [g, g], color=common.INK, lw=1.5)
        ax.text(i + 0.3, g, f"{g:.2f}×", va="center", fontsize=6.5)
    ax.set(yscale="log", ylabel="speed-up vs always-GPU", xlim=(-0.5, 3.8))
    ax.set_xticks(range(4), [names[c] for c in conds])
    style(ax)
    save(fig, "rq2")


def hybrid():
    """Hybrid gain over GPU-only by batch size."""
    rows = read("hybrid_gain.csv")
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    ax.axhline(0, color=common.MUTED, lw=0.8, ls="--")
    markers = "os^vD<>ph*"
    for op, m in zip(dict.fromkeys(r["op"] for r in rows), markers):
        p = [r for r in rows if r["op"] == op]
        ax.plot([int(r["batch"]) for r in p], [100 * float(r["gain"]) for r in p], marker=m, ms=3, lw=1, label=op)
    ax.set(xscale="log", xlabel="batch (rows for filter)", ylabel="Hybrid gain vs GPU (%)")
    ax.legend(frameon=False, ncol=5, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    style(ax)
    save(fig, "hybrid")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for draw in (crossover, accuracy, confidence, rq2, hybrid):
        draw()
    print("wrote", OUT_DIR)
