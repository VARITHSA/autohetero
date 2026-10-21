"""RQ2 (DESIGN.md §1, §10): does rebalancing recover time under contention? Plus mean speedups and variance.

    python eval/rq2.py                       # newest bench/results/*.csv -> eval/out/rq2.csv, rq2_summary.csv, rq2.png

From bench/experiments.py's rq2 rows, per condition and workload (warm-up rep 0 left out, except first_call):
  the median time of each strategy; the oracle, the best measured option (CPU, GPU or Hybrid) in each rep;
  AutoHetero's speedup over GPU-only and CPU-only, its gap to the oracle, its spread across runs (CV), how
  often it switched to Hybrid, how long it took to notice (detect), and its first call (decisions, trials).
The per-condition summary uses geometric means across workloads. rq2.csv is the figure's table view.
"""
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

try:
    from eval import common
except ImportError:  # run as a script: eval/ itself is on the path
    import common

OPTIONS = ("cpu", "gpu", "hybrid")  # what the oracle chooses from
PLOTTED = ("cpu", "gpu", "autohetero")
LABELS = {"cpu": "CPU only", "gpu": "GPU only", "autohetero": "AutoHetero"}


def load(path):
    """{(condition, workload, size): {strategy: {rep: row}}} for the rq2 experiment."""
    runs = defaultdict(lambda: defaultdict(dict))
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["experiment"] == "rq2":
                runs[r["condition"], r["workload"], r["size"]][r["strategy"]][int(r["rep"])] = r
    return runs


def table(runs):
    rows = []
    for (condition, workload, size), by in runs.items():
        reps = sorted(rep for rep in by["autohetero"] if rep > 0 and all(rep in by[s] for s in by))
        if not reps or not {"cpu", "gpu"} <= set(by):  # the run died before this workload finished a rep
            print(f"skipped incomplete workload: {condition} {workload} {size}")
            continue
        seconds = {s: [float(by[s][rep]["seconds"]) for rep in reps] for s in by}
        oracle = [min(float(by[s][rep]["seconds"]) for s in OPTIONS if s in by) for rep in reps]
        auto_runs = [by["autohetero"][rep] for rep in reps]
        switched = [r for r in auto_runs if r["slowdown"]]
        med = {s: statistics.median(t) for s, t in seconds.items()}
        auto, best = med["autohetero"], statistics.median(oracle)
        rows.append({
            "condition": condition, "workload": workload, "size": size,
            **{f"{s}_ms": med[s] * 1e3 for s in ("cpu", "gpu", "hybrid", "autohetero") if s in med},
            "oracle_ms": best * 1e3,
            "speedup_vs_gpu": med["gpu"] / auto, "speedup_vs_cpu": med["cpu"] / auto, "gap_to_oracle": auto / best - 1,
            "autohetero_cv": statistics.stdev(seconds["autohetero"]) / auto if len(reps) > 1 else 0.0,
            "autohetero_modes": " ".join(sorted({r["mode"] for r in auto_runs})),
            "switch_rate": len(switched) / len(auto_runs),
            "detect_ms": statistics.median(float(r["detect_seconds"]) for r in switched) * 1e3 if switched else "",
            "chunks_before_switch": statistics.median(int(r["chunks_before_switch"]) for r in switched) if switched else "",
            "first_call_ms": float(by["autohetero"][0]["seconds"]) * 1e3 if 0 in by["autohetero"] else "",
        })
    return rows


def summary(rows):
    out = []
    for condition in dict.fromkeys(r["condition"] for r in rows):
        mine = [r for r in rows if r["condition"] == condition]
        geomean = lambda xs: math.exp(statistics.mean(math.log(x) for x in xs))
        detects = [r["detect_ms"] for r in mine if r["detect_ms"] != ""]
        out.append({
            "condition": condition, "workloads": len(mine),
            "speedup_vs_gpu_geomean": geomean(r["speedup_vs_gpu"] for r in mine),
            "speedup_vs_cpu_geomean": geomean(r["speedup_vs_cpu"] for r in mine),
            "gap_to_oracle_geomean": geomean(1 + r["gap_to_oracle"] for r in mine) - 1,
            "within_5pct_of_oracle": sum(r["gap_to_oracle"] <= 0.05 for r in mine) / len(mine),
            "switch_rate": statistics.mean(r["switch_rate"] for r in mine),
            "detect_ms_median": statistics.median(detects) if detects else "",
            "autohetero_cv_median": statistics.median(r["autohetero_cv"] for r in mine),
        })
    return out


def plot(rows, out, subtitle):
    """Per condition: every workload's time relative to the oracle (1 = best measured option), per strategy."""
    conditions = list(dict.fromkeys(r["condition"] for r in rows))
    workloads = list(dict.fromkeys((r["workload"], r["size"]) for r in rows))
    height = 0.28 * len(workloads) + 1.6
    fig, axes = plt.subplots(1, len(conditions), figsize=(4.2 * len(conditions) + 2.2, height), sharey=True,
                             squeeze=False, facecolor=common.SURFACE)
    for ax, condition in zip(axes[0], conditions):
        common.style(ax)
        by = {(r["workload"], r["size"]): r for r in rows if r["condition"] == condition}
        for y, w in enumerate(workloads):
            for s in PLOTTED:
                if w in by and f"{s}_ms" in by[w]:
                    ax.plot(by[w][f"{s}_ms"] / by[w]["oracle_ms"], y, "o", ms=6, color=common.COLORS[s],
                            mec=common.SURFACE, mew=1.5, zorder=3)
        ax.axvline(1, color=common.AXIS, lw=1, zorder=1)
        ax.set_xscale("log")
        ax.set_title(condition, loc="left", fontsize=10, color=common.INK)
        ax.set_xlabel("time relative to the oracle (1 = best)", fontsize=8, color=common.MUTED)
    axes[0][0].set_yticks(range(len(workloads)), [f"{w} {s}" for w, s in workloads])
    axes[0][0].invert_yaxis()
    handles = [Line2D([], [], ls="", marker="o", ms=6, color=common.COLORS[s], label=LABELS[s]) for s in PLOTTED]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=9, labelcolor=common.INK)
    common.header(fig, "RQ2: time per workload under each condition, relative to the oracle (median runs)", subtitle)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(data, out_dir):
    rows = table(load(data))
    out_dir.mkdir(parents=True, exist_ok=True)
    common.write_csv(rows, out_dir / "rq2.csv")
    summary_rows = summary(rows)
    common.write_csv(summary_rows, out_dir / "rq2_summary.csv")
    plot(rows, out_dir / "rq2.png", common.machine(data))
    print(f"{'condition':10s} {'vs GPU':>8s} {'vs CPU':>8s} {'gap to oracle':>14s} {'<=5% gap':>9s} {'switches':>9s}")
    for s in summary_rows:
        print(f"{s['condition']:10s} {s['speedup_vs_gpu_geomean']:7.2f}x {s['speedup_vs_cpu_geomean']:7.2f}x "
              f"{s['gap_to_oracle_geomean']:13.1%} {s['within_5pct_of_oracle']:9.0%} {s['switch_rate']:9.0%}")
    return out_dir / "rq2.png"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", nargs="?", help="experiment results CSV (default: newest in bench/results)")
    parser.add_argument("--out", type=Path, default=common.OUT)
    a = parser.parse_args()
    data = a.data or common.newest("bench/results")
    if not data:
        parser.error("no results: run bench/experiments.py first, or pass a CSV path")
    print("wrote", main(data, a.out))
