"""AutoHetero demo (DESIGN.md §10).

    streamlit run demo/app.py

Runs demo/pipeline.py's steps three ways: as written (AutoHetero decides each step), and again under
force("cpu") and force("gpu") for the CPU-only and GPU-only columns. The GPU stress switch starts a parallel
PyTorch GPU job (bench/stress.py), so you can watch AutoHetero rebalance.
"""
import contextlib
import io
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for bench/stress.py

import autohetero  # noqa: E402
from autohetero import devices, history  # noqa: E402
from bench import stress  # noqa: E402

RESOLUTIONS = {"256 x 256": (256, 256), "720p": (720, 1280)}
VARIANTS = {"AutoHetero": None, "CPU only": "cpu", "GPU only": "gpu"}


def run(images, mode):
    """[(step, seconds, mode it ran in)] for one pass of the pipeline; mode None lets AutoHetero decide."""
    rows, x = [], images
    with autohetero.force(mode) if mode else contextlib.nullcontext():
        for name, step in pipeline.STEPS.items():
            start = time.perf_counter()
            x = step(x)
            r = history.last()
            ran = r["mode"].upper() + (" -> HYBRID" if r["slowdown"] else "")  # a slowdown switched it mid-call
            rows.append((name, time.perf_counter() - start, ran))
    return rows


st.set_page_config(page_title="AutoHetero demo", layout="wide")
st.title("AutoHetero demo")
st.caption("The same four steps, run three ways. The app's only AutoHetero line is `autohetero.enable()` (demo/pipeline.py).")

with st.spinner("Calibrating this machine (first run only, about 30 s)..."):
    import pipeline  # noqa: E402  (calls autohetero.enable())

if devices.GPU is None:
    st.warning("No NVIDIA GPU found: every column runs on the CPU.")

with st.sidebar:
    batch = st.select_slider("Images per batch", options=[1, 8, 32, 64], value=32)
    resolution = st.selectbox("Resolution", list(RESOLUTIONS), index=1)
    stressed = st.toggle("GPU stress (a parallel GPU job)", disabled=devices.GPU is None)
    go = st.button("Run", type="primary")

if stressed and "stress" not in st.session_state:  # a context manager kept open across Streamlit reruns
    st.session_state.stress = stress.gpu(device=str(devices.GPU))
    st.session_state.stress.__enter__()
elif not stressed and "stress" in st.session_state:
    st.session_state.pop("stress").__exit__(None, None, None)

if go:
    images = torch.rand(batch, 3, *RESOLUTIONS[resolution], generator=torch.Generator().manual_seed(0))
    if "warm" not in st.session_state:  # build the model and copy it to each device before timing anything
        for mode in VARIANTS.values():
            run(images[:2], mode)
        st.session_state.warm = True
    log = io.StringIO()
    results = {}
    for variant, mode in VARIANTS.items():
        with contextlib.redirect_stderr(log) if mode is None else contextlib.nullcontext():
            results[variant] = run(images, mode)

    best_baseline = min(sum(s for _, s, _ in results[v]) for v in ("CPU only", "GPU only"))
    for column, (variant, rows) in zip(st.columns(3), results.items()):
        total = sum(s for _, s, _ in rows)
        delta = None if variant != "AutoHetero" else f"{(total - best_baseline) / best_baseline:+.0%} vs best single device"
        column.metric(variant, f"{total * 1e3:.0f} ms", delta, delta_color="inverse")
        column.dataframe(pd.DataFrame(rows, columns=["step", "seconds", "ran on"]).assign(
            ms=lambda d: (d.seconds * 1e3).round(1))[["step", "ms", "ran on"]], hide_index=True)
    st.subheader("Decision log")
    st.code(log.getvalue() or "(no new decisions: every size was decided before)", language=None)
    st.subheader("autohetero.report(), this session")
    with contextlib.redirect_stdout(io.StringIO()):
        report = autohetero.report()
    st.dataframe(pd.DataFrame(report), hide_index=True)
    st.caption("Counts include the forced CPU-only and GPU-only reruns. saved_vs_gpu_estimate is an estimate: "
               "predicted always-GPU time minus measured time, over the calls AutoHetero decided.")
