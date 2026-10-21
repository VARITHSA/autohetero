"""Crossover graphs (DESIGN.md §11, weeks 3-4 exit check): CPU vs GPU time per op, and where the CPU is faster.

    python eval/crossover.py                      # newest bench/data/*.csv -> eval/out/crossover.png and .csv
    python eval/crossover.py bench/data/<file>.csv

The .csv next to the figure is its table view: per op, the sizes where each device wins.
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from autohetero.ops import inputs  # noqa: E402

try:
    from eval import common
except ImportError:  # run as a script: eval/ itself is on the path
    import common

CPU_REGION = "#cde2fb"  # lightest step of the CPU's blue ramp


def load(path):
    """{op: {size: (input elements, {device: median seconds})}}"""
    runs = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            runs[r["op"], r["size"], int(r["elements"]), r["device"]].append(float(r["seconds"]))
    medians = defaultdict(dict)
    for (op, size, elements, device), times in runs.items():
        medians[op].setdefault(size, (elements, {}))[1][device] = statistics.median(times)
    return medians


def summarize(medians):
    """One row per op: where each device wins, and the GPU's speedup range (CPU time / GPU time)."""
    rows = []
    for op, sizes in medians.items():
        both = [(e, t["cpu"], t["gpu"]) for e, t in sizes.values() if len(t) == 2]
        speedups = [c / g for _, c, g in both]
        rows.append({
            "op": op,
            "sizes_on_both": len(both),
            "cpu_faster_up_to_elements": max((e for e, c, g in both if c < g), default=""),
            "gpu_faster_from_elements": min((e for e, c, g in both if g < c), default=""),
            "gpu_speedup_min": f"{min(speedups):.3g}" if speedups else "",
            "gpu_speedup_max": f"{max(speedups):.3g}" if speedups else "",
        })
    return rows


def cpu_faster_spans(both):
    """x-ranges to shade: each size where the CPU wins, out to the geometric midpoints with its neighbours."""
    both = sorted(both)
    xs = [e for e, _, _ in both]
    edges = [xs[0]] + [math.sqrt(a * b) for a, b in zip(xs, xs[1:])] + [xs[-1]] if xs else []
    return [(edges[i], edges[i + 1]) for i, (_, c, g) in enumerate(both) if c < g]


def plot(medians, out, subtitle):
    names = [op for op in inputs.OPS if op in medians]
    rows = math.ceil(len(names) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(10, 2.6 * rows + 1.2), squeeze=False, facecolor=common.SURFACE)
    for ax, op in zip(axes.flat, names):
        sizes = medians[op].values()
        for lo, hi in cpu_faster_spans([(e, t["cpu"], t["gpu"]) for e, t in sizes if len(t) == 2]):
            ax.axvspan(lo, hi, color=CPU_REGION, lw=0, zorder=0)
        # One line per batch sweep: a line only joins sizes that differ in dim 0 (image ops sweep 5 resolutions).
        lines = defaultdict(list)
        for size, (e, t) in medians[op].items():
            shape = ast.literal_eval(size)
            for device, seconds in t.items():
                lines[device, shape[1:] if isinstance(shape, tuple) else ()].append((e, seconds * 1e3))
        for (device, _), points in lines.items():
            ax.plot(*zip(*sorted(points)), color=common.COLORS[device], lw=2, marker="o", ms=5, mec=common.SURFACE, mew=1, zorder=2)
        common.style(ax)
        ax.set(xscale="log", yscale="log")
        ax.set_title(op, loc="left", fontsize=10, color=common.INK)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("median time (ms)", fontsize=8, color=common.MUTED)
    for ax in axes[-1]:
        ax.set_xlabel("input elements", fontsize=8, color=common.MUTED)
    handles = [Line2D([], [], color=common.COLORS[d], lw=2, marker="o", ms=5, label=d.upper()) for d in ("cpu", "gpu")]
    handles.append(Patch(color=CPU_REGION, label="CPU faster"))
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=9, labelcolor=common.INK)
    common.header(fig, "CPU vs GPU time per op (median of 5 runs, transfers included)", subtitle)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(data, out_dir):
    medians = load(data)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot(medians, out_dir / "crossover.png", common.machine(data))
    rows = summarize(medians)
    common.write_csv(rows, out_dir / "crossover.csv")
    for r in rows:
        print("  ".join(f"{k}={v}" for k, v in r.items()))
    return out_dir / "crossover.png"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", nargs="?", help="benchmark CSV (default: newest in bench/data)")
    parser.add_argument("--out", type=Path, default=common.OUT)
    a = parser.parse_args()
    data = a.data or common.newest("bench/data")
    if not data:
        parser.error("no benchmark data: run bench/collect.py first, or pass a CSV path")
    print("wrote", main(data, a.out))
