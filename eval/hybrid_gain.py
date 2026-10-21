"""Hybrid gain (DESIGN.md §10): on an idle machine, how much faster is Hybrid than GPU-only, by batch size?

    python eval/hybrid_gain.py               # newest bench/results/*.csv -> eval/out/hybrid_gain.csv, _summary.csv, .png

gain = (GPU-only - Hybrid) / GPU-only, from median times (warm-up rep 0 left out); positive means Hybrid is
faster. The deck's 10-30% is a hypothesis: small batches are expected to show none. "gain_from_batch" is the
smallest batch from which every larger batch measured shows a gain. Only Hybrid runs where both devices did
work count: a single item can't split, and a speed-based split can give one device nothing; either way the run
measures CPU vs GPU, not the gain from using both. The figure plots the speed ratio
GPU / Hybrid on a log scale instead, so a 70x loss and a 1.2x gain both stay readable.
"""
import argparse
import ast
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

try:
    from eval import common
except ImportError:  # run as a script: eval/ itself is on the path
    import common


def table(path):
    runs = defaultdict(lambda: defaultdict(list))  # (op, size) -> strategy -> seconds
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["experiment"] == "hybrid_gain" and int(r["rep"]) > 0 and (r["strategy"] == "gpu" or _both(r)):
                runs[r["workload"], r["size"]][r["strategy"]].append(float(r["seconds"]))
    rows = []
    for (op, size), by in runs.items():
        if not (by["gpu"] and by["hybrid"]):  # the run died before this size finished, or Hybrid couldn't split it
            print(f"skipped (no run used both devices): {op} {size}")
            continue
        shape = ast.literal_eval(size)
        gpu, hybrid = statistics.median(by["gpu"]), statistics.median(by["hybrid"])
        rows.append({"op": op, "size": size, "batch": shape[0] if isinstance(shape, tuple) else shape,
                     "gpu_ms": gpu * 1e3, "hybrid_ms": hybrid * 1e3, "gain": (gpu - hybrid) / gpu})
    return sorted(rows, key=lambda r: (r["op"], r["batch"]))


def _both(r):
    """Whether a Hybrid run really used both devices (files from before gpu_share was recorded: mode only)."""
    share = r.get("gpu_share") or ""
    return r["mode"] == "hybrid" and (share == "" or 0 < float(share) < 1)


def summary(rows):
    out = []
    for op in dict.fromkeys(r["op"] for r in rows):
        mine = [r for r in rows if r["op"] == op]
        losing = [r["batch"] for r in mine if r["gain"] <= 0]
        start = [r["batch"] for r in mine if r["gain"] > 0 and r["batch"] > max(losing, default=-1)]
        best = max(mine, key=lambda r: r["gain"])
        out.append({"op": op, "gain_from_batch": min(start) if start else "none", "max_gain": best["gain"],
                    "max_gain_batch": best["batch"]})
    return out


def plot(rows, out, subtitle):
    ops = list(dict.fromkeys(r["op"] for r in rows))
    cols = min(4, len(ops))
    grid = math.ceil(len(ops) / cols)
    fig, axes = plt.subplots(grid, cols, figsize=(3.3 * cols, 2.5 * grid + 1.2), squeeze=False, facecolor=common.SURFACE)
    for ax, op in zip(axes.flat, ops):
        common.style(ax)
        points = [(r["batch"], r["gpu_ms"] / r["hybrid_ms"]) for r in rows if r["op"] == op]
        ax.plot(*zip(*points), color=common.COLORS["hybrid"], lw=2, marker="o", ms=5, mec=common.SURFACE, mew=1)
        ax.axhline(1, color=common.AXIS, lw=1)
        ax.set(xscale="log", yscale="log")
        ax.minorticks_off()
        step = 2 if len(points) > 6 else 1  # crowded panels label every other batch
        ax.set_xticks([b for b, _ in points], [_compact(b) if i % step == 0 else "" for i, (b, _) in enumerate(points)])
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))  # 0.1x 0.2x 0.5x 1x 2x ...
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}x"))
        ax.set_title(op, loc="left", fontsize=10, color=common.INK)
    for ax in axes.flat[len(ops):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Hybrid speed vs GPU-only", fontsize=8, color=common.MUTED)
    for ax in axes[-1]:
        ax.set_xlabel("batch (rows, or matrix side)", fontsize=8, color=common.MUTED)
    common.header(fig, "Hybrid vs GPU-only on an idle machine (above 1x: Hybrid faster)", subtitle)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _compact(n):
    return f"{n:g}" if n < 10_000 else f"{n:.0e}".replace("e+0", "e")  # 1000000 -> 1e6


def main(data, out_dir):
    rows = table(data)
    out_dir.mkdir(parents=True, exist_ok=True)
    common.write_csv(rows, out_dir / "hybrid_gain.csv")
    summary_rows = summary(rows)
    common.write_csv(summary_rows, out_dir / "hybrid_gain_summary.csv")
    plot(rows, out_dir / "hybrid_gain.png", common.machine(data))
    for s in summary_rows:
        print(f"{s['op']:12s} gain from batch {str(s['gain_from_batch']):>8s}   max {s['max_gain']:6.1%} at batch {s['max_gain_batch']}")
    return out_dir / "hybrid_gain.png"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", nargs="?", help="experiment results CSV (default: newest in bench/results)")
    parser.add_argument("--out", type=Path, default=common.OUT)
    a = parser.parse_args()
    data = a.data or common.newest("bench/results")
    if not data:
        parser.error("no results: run bench/experiments.py first, or pass a CSV path")
    print("wrote", main(data, a.out))
