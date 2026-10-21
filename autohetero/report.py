"""autohetero.report(): what ran where this session, how long it took, and the estimated time saved (DESIGN.md §9)."""
from . import history

NOTE = ("saved vs GPU is an ESTIMATE: predicted always-GPU time minus measured time, summed over calls that had "
        "a prediction (forced calls and first measurements have none)")


def report():
    """Print one row per dispatched function this session, and return the rows as dicts.

    Columns: calls, modes chosen (with counts), total measured time, estimated time saved vs always-GPU,
    fallbacks to the CPU after a GPU error, and switches to Hybrid after a slowdown.
    """
    rows = [
        {
            "function": name.removeprefix("__main__."),
            "calls": int(t["calls"]),
            "modes": ", ".join(f"{k.removeprefix('mode_').upper()} {int(n)}" for k, n in t.items() if k.startswith("mode_")),
            "seconds": t["seconds"],
            "saved_vs_gpu_estimate": t["saved_vs_gpu_estimate"] if t["predicted_calls"] else None,
            "fallbacks": int(t["fallbacks"]),
            "switches": int(t["switches"]),
        }
        for name, t in history.totals().items()
    ]
    rows.sort(key=lambda r: r["function"])
    if not rows:
        print("[AutoHetero] no dispatched calls yet")
        return rows
    estimates = [r["saved_vs_gpu_estimate"] for r in rows if r["saved_vs_gpu_estimate"] is not None]
    total = {"function": "total", "calls": sum(r["calls"] for r in rows), "modes": "",
             "seconds": sum(r["seconds"] for r in rows), "saved_vs_gpu_estimate": sum(estimates) if estimates else None,
             "fallbacks": sum(r["fallbacks"] for r in rows), "switches": sum(r["switches"] for r in rows)}
    header = ("function", "calls", "modes", "time", "saved vs GPU (est.)", "fallbacks", "switches")
    lines = [[r["function"], str(r["calls"]), r["modes"], _time(r["seconds"]),
              "n/a" if r["saved_vs_gpu_estimate"] is None else _time(r["saved_vs_gpu_estimate"]),
              str(r["fallbacks"]), str(r["switches"])] for r in [*rows, total]]
    widths = [max(len(c) for c in column) for column in zip(header, *lines)]
    fmt = lambda cells: "  ".join(c.ljust(w) if i in (0, 2) else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths)))
    print("\n".join([fmt(header), fmt(["-" * w for w in widths]), *map(fmt, lines[:-1]), fmt(["-" * w for w in widths]), fmt(lines[-1]), NOTE]))
    return rows


def _time(seconds):
    return f"{seconds:.2f} s" if abs(seconds) >= 1 else f"{seconds * 1e3:.1f} ms"
