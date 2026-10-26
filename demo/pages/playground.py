"""AutoHetero playground: run any built-in op at any size and see where AutoHetero runs it.

    streamlit run demo/app.py        # then "playground" in the sidebar

The op runs once with AutoHetero deciding, then once with each option forced, so the decision can be checked
against what each option really takes. The snippet at the bottom makes the same call in your own code.
"""
import contextlib
import math
import time

import pandas as pd
import streamlit as st
import torch

import autohetero
from autohetero import calibrate, devices, history, ops
from autohetero.ops import inputs

# CPU last: a GPU run right after a long CPU run is slowed (CARRYOVER.md, row 12)
STRATEGIES = {"AutoHetero": None, "GPU only": "gpu", "Hybrid": "hybrid", "CPU only": "cpu"}


def label(op, size):
    if op == "matmul":
        return f"{size} × {size} matrices"
    if isinstance(size, int):
        return f"{size:,} rows"
    return " × ".join(map(str, size))


def snippet(op, size):
    """The same call in plain PyTorch, ready to paste."""
    if op in ("filter", "sort"):
        make, args = f"rows = torch.rand({size}, 4)", "rows"
    elif op == "groupby":
        make, args = f"keys = torch.randint(0, {inputs.GROUPS}, ({size},))\nvalues = torch.rand({size})", "keys, values"
    elif op == "matmul":
        make, args = f"a = torch.rand({size}, {size})\nb = torch.rand({size}, {size})", "a, b"
    else:
        make, args = f"x = torch.rand{size}", "x"
    return (f"import torch\nimport autohetero\n\nautohetero.enable()  # once: the first time, it calibrates this machine\n"
            f"{make}\nout = autohetero.ops.{op}({args})  # runs where it's predicted fastest\nautohetero.report()  # what ran where")


def timed(fn, args, mode):
    """(seconds, history record) for one call; mode None lets AutoHetero decide."""
    with autohetero.force(mode) if mode else contextlib.nullcontext():
        start = time.perf_counter()
        fn(*args)
        return time.perf_counter() - start, history.last()


def ran_on(record):
    where = record["mode"].upper() + (" → HYBRID" if record["slowdown"] else "")  # a slowdown switched it mid-call
    return where + (" (GPU failed, ran on CPU)" if record["fallback"] else "")


st.set_page_config(page_title="AutoHetero playground", layout="wide")
st.title("AutoHetero playground")
st.caption("Pick a built-in op and an input size. AutoHetero decides where to run it; the forced runs show what each option really takes.")

with st.spinner("Calibrating this machine (first run only, about 30 s)..."):
    autohetero.enable()
    info = calibrate.calibration()

with st.container(border=True):
    if devices.GPU is None:
        st.warning("No NVIDIA GPU found: AutoHetero runs everything on the CPU.")
    else:
        gpu = str(devices.GPU)
        if devices.GPU.type == "cuda":
            gpu = f"{torch.cuda.get_device_name(devices.GPU)} ({torch.cuda.get_device_properties(devices.GPU).total_memory / 2**30:.0f} GB)"
        columns = st.columns(4)
        columns[0].metric("GPU", gpu)
        columns[1].metric("Calibration (once)", f"{info['seconds']} s")
        columns[2].metric("CPU → GPU", f"{info['h2d_bytes_per_s'] / 1e9:.1f} GB/s")
        columns[3].metric("GPU → CPU", f"{info['d2h_bytes_per_s'] / 1e9:.1f} GB/s")
    st.caption(f"CPU: {calibrate.machine()['cpu']}")

left, right = st.columns([1, 2])
op = left.selectbox("Op", inputs.OPS, index=inputs.OPS.index("sobel"))
fit = [s for series in inputs.sizes(op) for s in series if calibrate.fits(op, s, devices.GPU or devices.CPU)]
fit.sort(key=lambda s: math.prod(s) if isinstance(s, tuple) else s)
if not fit:
    st.error(f"No size of {op} fits in free memory right now.")
    st.stop()
# by position: the slider would read a tuple value (an image shape) as a range; one slider state per op
size = fit[right.select_slider("Input size (only sizes that fit in free memory)", range(len(fit)), value=len(fit) // 2,
                               format_func=lambda i: label(op, fit[i]), key=f"size-{op}")]
go = st.button("Run", type="primary")
st.caption("The first run of each size runs every option once untimed, as a warm-up. Large sizes can take a while on the CPU.")

if go:
    strategies = {k: m for k, m in STRATEGIES.items() if devices.GPU is not None or m in (None, "cpu")}
    if op not in ops.SPLITTABLE:
        strategies.pop("Hybrid", None)  # only splittable calls can be shared between the devices
    fn, args = getattr(ops, op), inputs.make(op, size, torch.Generator().manual_seed(0))
    with st.spinner("Running..."):
        warm = st.session_state.setdefault("warm", set())
        if (op, size) not in warm:  # first calls pay one-off costs: copying a model, loading GPU kernels, deciding
            for mode in strategies.values():
                timed(fn, args, mode)
            warm.add((op, size))
        results = {name: timed(fn, args, mode) for name, mode in strategies.items()}

    seconds, record = results["AutoHetero"]
    fastest = min(s for name, (s, _) in results.items() if name != "AutoHetero") if len(results) > 1 else seconds
    predicted = " · ".join(f"{m.upper()} {t * 1e3:.1f} ms" for m, t in sorted(record["predicted"].items(), key=lambda kv: kv[1]))
    st.metric(f"AutoHetero ran it on {ran_on(record)}", f"{seconds * 1e3:.1f} ms",
              f"{(seconds - fastest) / fastest:+.0%} vs the fastest forced option", delta_color="inverse")
    if predicted:
        st.caption(f"Predicted before running: {predicted}")
    table = pd.DataFrame([(name, s * 1e3, ran_on(r)) for name, (s, r) in results.items()], columns=["option", "ms", "ran on"])
    table["fastest"] = (table.ms == table.ms.min()).map({True: "←", False: ""})
    st.dataframe(table.round({"ms": 1}), hide_index=True)

    st.subheader("Use it in your code")
    st.code(snippet(op, size), language="python")
