# AutoHetero — Implementation Requirements

Taken from `DESIGN.md` (2026-10-20). Each requirement has an ID that stays fixed, so cite it in commits and PRs. **Src** is the DESIGN.md section; **Wk** is the §11 phase in which it gets built.
The targets in §1 (≥ 90%, within 5%, 10–30%, < 1 ms) are hypotheses: measure and report them, but never assert or test against them.
Rows marked *(inferred)* are my reading where DESIGN.md is silent. Confirm them or change them as a team (see "Open questions" at the end).

---

## C — Environment and conventions

| ID | Requirement | Src | Wk |
|----|-------------|-----|----|
| C-1 | Use Python ≥ 3.11 (the dev Mac runs 3.14), PyTorch 2.x, torchvision (for model definitions, random weights), NumPy, scikit-learn, psutil, nvidia-ml-py and matplotlib, plus Streamlit for the demo only. Any other dependency needs a stated reason. | §12 | 1 |
| C-2 | `pyproject.toml` must support `pip install -e .`, with the package named `autohetero`. Follow the repo layout in §4. | §4 | 1 |
| C-3 | Fix seeds everywhere: data generation, forests and splits. | §12 | 1 |
| C-4 | `pytest` passes on a CPU-only machine. GPU paths are tested through a fake device, and real-GPU tests are marked `@pytest.mark.gpu`. | §12 | 1 |
| C-5 | Every shortcut carries a `# shortcut: <the limit>, <when to upgrade>` comment. | §12 | all |
| C-6 | `calibrate.py`, `features.py`, `predictor.py`, `monitor.py` and `bench/` contain no decorator or dispatch code, so they can be lifted out as a library. | §13 | 3–5 |

## API — Public surface (`__init__.py`)

| ID | Requirement | Src | Wk |
|----|-------------|-----|----|
| API-1 | The public surface is exactly `enable()`, `auto`, `force(mode)` and `report()` (but see Q1 about `retrain()`). | §2 | 5 |
| API-2 | `enable()` turns on dispatch for every `@auto` function and every built-in `autohetero.ops` op. It does **not** monkey-patch torch. | §2 | 5 |
| API-3 | Support both `@auto` and `@auto(splittable=True, batch_dim=0)`. The default is `splittable=False`. Document that `@auto` functions must not modify their inputs in place. | §2 | 5 |
| API-4 | `force("cpu" \| "gpu" \| "hybrid")` is a context manager backed by a `contextvars` variable, and it skips prediction. Any other value is rejected *(inferred)*. | §2, §5 | 5 |
| API-5 | Accept `torch.Tensor` and `numpy.ndarray` inputs. Return outputs in the same type and on the same device as the inputs, so the user never writes `.to("cuda")`. | §2 | 1–2 |
| API-6 | Move `nn.Module` arguments too. Keep one cached copy per device (weights copied once, not on every call). | §2 | 1–2 |
| API-7 | Document that a model captured in a closure or a global can't be moved. Built-in model ops manage their own copies. | §2 | 5 |
| API-8 | Nesting: the outermost dispatched call picks the device, and built-in ops called inside it run on that device without dispatching again. | §2 | 5 |
| API-9 | Decision log: one line for each new decision and each mode switch, plus calibration start and finish. Formats: `[AutoHetero] enhance: batch 200, GPU`, `[AutoHetero] Slowdown 1.8x, Hybrid`. `AUTOHETERO_QUIET=1` silences it. | §2 | 5–6 |

## DSP — Call path (`dispatch.py`)

| ID | Requirement | Src | Wk |
|----|-------------|-----|----|
| DSP-1 | Order of steps: force check → features → decide → execute → return in the caller's type and device → record in history. | §5 | 5 |
| DSP-2 | Safe fallback: any GPU error (no CUDA, OOM, driver error) reruns the call on CPU, marks the GPU unhealthy for that (function, size bucket) for the rest of the session, and logs once. **User code never crashes because of AutoHetero.** | §5 | 5 |
| DSP-3 | Document that a function with side effects may run twice on fallback. | §5 | 5 |
| DSP-4 | With no GPU: skip calibration, run everything on CPU, and raise no errors. | §6 | 5 |

## DEV — Devices (`devices.py`)

| ID | Requirement | Src | Wk |
|----|-------------|-----|----|
| DEV-1 | Move inputs to the chosen device and outputs back, converting numpy↔torch both ways. | §2, §4 | 1–2 |
| DEV-2 | Keep a per-device model cache (see API-6). | §2 | 1–2 |
| DEV-3 | Provide the CPU fallback path that DSP-2 uses. | §4 | 1–2 |

## OPS — Built-in ops (`ops/`)

All ops are written in PyTorch with one code path that runs on either device.

| Op | Domain | Sizes | Splittable *(inferred unless noted)* |
|----|--------|-------|------------|
| blur (Gaussian), resize, sobel | Image | small → 4K, batch 1–256 | yes, per image |
| resnet18, mobilenetv2 | Inference | batch 1–256 | yes, per sample |
| filter | Analytics | 10³–10⁷ rows | yes, per row (concatenate in order) |
| sort, groupby | Analytics | 10³–10⁷ rows | **no** (stated in §2) |
| matmul | Scientific | log-spaced up to the GPU memory limit | yes, by rows of A |
| fft | Signal | log-spaced up to the GPU memory limit | only along a batch of independent signals |

| ID | Requirement | Src | Wk |
|----|-------------|-----|----|
| OPS-1 | Implement all 10 ops above. | §4, §10 | 1–2 |
| OPS-2 | **Exit check:** CPU and GPU results match within float tolerance at 3 sizes for every op. | §11 | 1–2 |
| OPS-3 | Model ops run in inference mode (`eval()`, no grad), since training is out of scope. | §3 | 1–2 |

## CAL — Calibration (`calibrate.py`)

| ID | Requirement | Src | Wk |
|----|-------------|-----|----|
| CAL-1 | Run once, on first use, in about 30 s. Measure host↔device bandwidth, kernel launch latency, and each built-in op on CPU and GPU at ~6 log-spaced sizes. | §6 | 3–4 |
| CAL-2 | Also measure the Hybrid `split_overhead`: thread start, chunk copy and concatenation. | §6 | 3–4 |
| CAL-3 | Cap each microbenchmark by **time**, not size, and measure the large CPU sizes at fewer points. Print progress. | §6 | 3–4 |
| CAL-4 | Cache results at `~/.cache/autohetero/<machine-key>/`, where the key is CPU model + GPU name + driver + torch version. Recalibrate when the key changes. | §6 | 3–4 |
| CAL-5 | No GPU → skip calibration. | §6 | 3–4 |

## FEA / PRD — Features and prediction (`features.py`, `predictor.py`)

| ID | Requirement | Src | Wk |
|----|-------------|-----|----|
| FEA-1 | Feature vector: op type (built-in name or `"user"`), input elements, input bytes, batch size, dtype, and CPU and GPU utilisation read from the monitor's cached sample. | §5 | 5 |
| PRD-1 | One `RandomForestRegressor` per device, trained on `log(time)` using the calibration and benchmark data. | §6 | 5 |
| PRD-2 | Train a `DecisionTree` alongside it for the RQ1 comparison. | §6 | 5 |
| PRD-3 | Keep the forest's `estimators_` reachable, not hidden behind a wrapper. | §13 | 5 |
| PRD-4 | Pick the mode with the fastest predicted time from CPU, GPU and Hybrid (Hybrid only for splittable calls). | §5, §6 | 5 |
| PRD-5 | Predicted Hybrid time = `T_cpu·T_gpu/(T_cpu+T_gpu) + split_overhead`. Pick Hybrid only when it beats the faster single device by a margin, and keep that margin in the constants file. | §6 | 6 |
| PRD-6 | Decision cache keyed by `(op, log2 size bucket, dtype, load bucket)`. Call the forest only on a cache miss, and invalidate an entry when online updates move its prediction. | §6 | 5 |
| PRD-7 | User functions: on the first call, run the first chunk on both devices (a trial), with the GPU chunk first because it works on a copy of the input. After that, use a per-function log-log fit of time vs. size for each device, updated online. | §6 | 6 |
| PRD-8 | Hybrid split: `gpu_share = s_gpu/(s_gpu+s_cpu)`, using measured or predicted speeds. | §6 | 6 |

## EXE — Execution (`executor.py`)

| ID | Requirement | Src | Wk |
|----|-------------|-----|----|
| EXE-1 | Run non-splittable calls and inputs below the minimum size directly. **Never chunk a non-splittable call.** | §5, §7 | 6 |
| EXE-2 | Split along `batch_dim`. The first chunk is ~10% of the batch but takes at least ~5 ms. | §7 | 6 |
| EXE-3 | After each chunk, compare the actual time with the predicted time. If actual > 1.5× predicted, switch to Hybrid and re-split by the measured speeds. Re-check after every round. | §7 | 6 |
| EXE-4 | Hybrid uses two threads: one feeds the GPU and one runs CPU ops. Each thread sees its own device for nested ops *(inferred, see Q4)*. | §7 | 6 |
| EXE-5 | Concatenate results in the original order. | §7 | 6 |
| EXE-6 | Keep all tuning constants in one module: 1.5×, 10%, 5 ms, minimum size, Hybrid margin, ~200 ms sample period. Tune them on benchmark data. | §7 | 6 |
| EXE-7 | **Exit check:** under GPU stress, a long task switches to Hybrid within 1–2 chunks. | §11 | 6 |

## MON / HIS — Monitoring and history (`monitor.py`, `history.py`)

| ID | Requirement | Src | Wk |
|----|-------------|-----|----|
| MON-1 | A daemon thread samples CPU utilisation (psutil) and GPU utilisation (NVML) every ~200 ms. Calls read the cached value, so the hot path makes no syscalls. | §8 | 3–4 |
| MON-2 | When NVML is missing (no NVIDIA GPU or driver), keep running with GPU utilisation unknown *(inferred)*. | §5 | 3–4 |
| HIS-1 | Append every measured call, with the **predicted** times logged next to the actual time. | §8, §13 | 6 |
| HIS-2 | Update per-function fits immediately. | §8 | 6 |
| HIS-3 | Retrain the forest only at idle or on `retrain()`, never inside a user call. | §8 | 6 |

## REP — Report (`report.py`)

| ID | Requirement | Src | Wk |
|----|-------------|-----|----|
| REP-1 | `report()` prints one row per function: mode chosen, calls, total time, time saved vs. always-GPU **labelled as an estimate** (from predicted times), fallback count and Hybrid-switch count. | §9 | 7 |

## TST — Must-have tests

| ID | Test | Src | Wk |
|----|------|-----|----|
| TST-1 | Hybrid output equals single-device output (see Q3 on tolerance). | §12 | 6 |
| TST-2 | Fallback works when CUDA raises. | §12 | 5 |
| TST-3 | `force()` is respected across threads. | §12 | 5 |
| TST-4 | numpy in → numpy out. | §12 | 1–2 |
| TST-5 | A non-splittable function is never chunked. | §12 | 6 |
| TST-6 | CPU/GPU parity for every op at 3 sizes (OPS-2), as a `gpu`-marked test. | §11 | 1–2 |
| TST-7 | On a GPU-less machine, everything runs on CPU with no errors. | §11 | 5 |

## BEN / EVL — Benchmarks and evaluation (`bench/`, `eval/`)

| ID | Requirement | Src | Wk |
|----|-------------|-----|----|
| BEN-1 | Run every workload × size × {CPU, GPU}: 5 repetitions after 1 discarded warm-up, timed **including transfers**, with `torch.cuda.synchronize()` before each GPU timer stops. | §10 | 3–4 |
| BEN-2 | Conditions: idle, GPU stress (a parallel PyTorch job), CPU stress (`stress-ng`) and battery saver. Battery saver needs an NVIDIA laptop; if nobody on the team has one, drop it and say so in the report. | §10 | 7 |
| BEN-3 | Baselines: always-CPU, always-GPU and an oracle (best measured option per run). | §10 | 7 |
| BEN-4 | Final numbers come from **one fixed machine**. Colab is for development only. | §10 | 7 |
| BEN-5 | **Exit check:** dataset collected, plus a crossover graph per op showing where CPU beats GPU. | §11 | 3–4 |
| EVL-1 | RQ1: prediction accuracy on an 80/20 split, k-fold CV and leave-one-op-out, forest vs. tree. | §10 | 5 |
| EVL-2 | RQ2: total time under contention vs. GPU-only, and recovery time after a slowdown. | §1, §10 | 7 |
| EVL-3 | RQ3: dispatch overhead per call, and lines changed in the demo app. | §1 | 5, 7 |
| EVL-4 | Hybrid vs. GPU-only across batch sizes for every splittable workload on an idle machine, showing where any gain starts. | §10 | 7 |
| EVL-5 | Metrics: mean speedup, prediction accuracy, recovery time, overhead per call, and variance across runs. | §10 | 7–8 |
| EVL-6 | Every AutoHetero number includes AutoHetero's own overhead: features, prediction, trials and chunking. | §10 | 7 |
| EVL-7 | Every graph and table comes from an `eval/` script that reads the collected data, with no hand-edited numbers. Replace every mock-up number in the deck. | §10, §11 | 8 |

## DEM — Demo (`demo/app.py`)

| ID | Requirement | Src | Wk |
|----|-------------|-----|----|
| DEM-1 | Streamlit app where integration takes **one added line**. Its CPU-only and GPU-only columns come from rerunning the same steps under `force("cpu")` and `force("gpu")`. | §10, §11 | 7 |

## CO — Phase 9 carry-over

| ID | Requirement | Src | Wk |
|----|-------------|-----|----|
| CO-1 | `CARRYOVER.md`: what was reused, what changed, and the §13 inference table filled with **measured** numbers. | §13 | 9 |
| CO-2 | Put the git tag `autohetero-v1` on the code the M.Tech work starts from. | §13 | 9 |
| CO-3 | Export the benchmark dataset in the M.Tech run-log format, marked as AutoHetero data that is never used as M.Tech evaluation data. | §13 | 9 |
| CO-4 | Cite AutoHetero as prior work with the team's agreement and with the guide informed. | §13 | 9 |
| CO-5 | Verify the 4 references before submission. | Refs | 8 |

---

## Phase → requirements

| Wk | Requirements | Exit check |
|----|--------------|------------|
| 1–2 | C-1…C-5, API-5, API-6, DEV-*, OPS-*, TST-4, TST-6 | OPS-2 |
| 3–4 | CAL-*, MON-*, BEN-1, BEN-5, C-6 | BEN-5 |
| 5 | API-1…4, 7…9, DSP-*, FEA-1, PRD-1…4, PRD-6, TST-2, TST-3, TST-7, EVL-1, EVL-3 | RQ1 reported, overhead measured, GPU-less run passes |
| 6 | PRD-5, PRD-7, PRD-8, EXE-*, HIS-*, TST-1, TST-5 | EXE-7 + TST-1 |
| 7 | REP-1, DEM-1, BEN-2…4, EVL-2, EVL-4…6 | All conditions × baselines measured |
| 8 | EVL-7, CO-5 | Every claim backed by a generated figure |
| 9 | CO-1…4 | Carry-over note + tag |

---

## Open questions (decide before week 5)

DESIGN.md leaves these open or contradicts itself. Each has my recommendation.

**Adopted in code so far (steps 13–16):** Q2 (`@auto` always dispatches; `enable()` adds the built-in ops and starts the monitor and calibration up front), Q4 (`force()` applies per thread or task and new threads don't inherit it; the Hybrid GPU thread runs on a copy of the caller's context), Q6 (the forest trains on this machine's calibration), Q8 (`force("hybrid")` on a non-splittable call runs on the faster single device), Q9 (the first array argument sets the output type and device; calls with no arrays run as written), and Q11 (built-ins split their first argument; user functions split every array argument). Since steps 17–18 also Q1 (`retrain()` is in the API) and Q5. For Q5, a splittable user function's first call times one ~10% piece on the GPU, then a different piece on the CPU, then runs Hybrid by the measured speeds. A non-splittable one follows the forest on its first call, runs on the other device on its second, then follows its fits. PRD-6 is met without any cache invalidation: user-function decisions come from their fits (microseconds), so they're never cached and can't go stale. Forest decisions are cleared on `retrain()`.

1. **`retrain()` vs. "keep exactly this surface".** §8 calls `autohetero.retrain()`, but §2 lists only four functions. → Add `retrain()` to §2.
2. **What does `enable()` add?** If `@auto` already dispatches by itself (Level 2), it's unclear what Level 1 changes. → `@auto` always dispatches; `enable()` turns on dispatch for built-in ops called at top level and starts the monitor and calibration up front.
3. **Hybrid "identical" to single-device.** CPU and GPU floats differ, so a result built partly on each device can't be bit-identical to a GPU-only result. → Use the same tolerance as OPS-2 in TST-1 and in the week-6 exit check. *Measured in step 3: even on the CPU alone, blurring a batch in 3 chunks differs from blurring it whole by up to 3.6e-7.* *Also: on Ampere-or-newer GPUs, PyTorch runs cuDNN convolutions in TF32 by default (about 1e-3 relative error), so the GPU part of a Hybrid result can differ from the CPU part by that much. The parity test turns TF32 off; TST-1 must either do the same or allow about 1e-3.*
4. **Context in Hybrid threads.** New `threading.Thread`s don't inherit `contextvars`, so without care, nested built-in ops inside Hybrid workers will dispatch again or see the wrong device. → Run each worker under `contextvars.copy_context()` with its own device set. TST-3 should also check isolation: a `force()` in one thread doesn't leak into another.
5. **Trial for non-splittable user functions.** There is no "first chunk", so a trial means running the whole call twice, which doubles both the cost and any side effects. → Run the first call on the predicted-safe device, run the second on the other device, then fit.
6. **Forest training data per machine.** No feature describes the machine, so a forest trained on the fixed machine's benchmark won't transfer to a user's machine, and on a new machine only ~6 sizes per op exist. Also, if all training data is idle, the utilisation features are constant and teach the model nothing. → Train the shipped forest on local calibration data plus local history at retrain time. Use the benchmark dataset for RQ1 evaluation. Either add stressed samples to the training set or leave load handling to the 1.5× check alone.
7. **"Within 5%"** (§1) isn't tied to any RQ or metric. → Define it (gap to oracle?) or drop it. *Settled 2026-10-25 (DESIGN §1): an RQ2 metric. AutoHetero's median time on a workload is at most 5% above the oracle's; `rq2_summary.csv` reports the share of workloads that meet it (`within_5pct_of_oracle`).*
8. **`force("hybrid")` on a non-splittable function.** Unspecified. → Log once and run on the faster single device, since crashing is not allowed.
9. **Return device for mixed or array-free inputs.** → The first tensor/array argument sets the return type and device. With no array arguments, run directly without dispatch.
10. **M.Tech run-log format** (CO-3) is undefined. → Get the schema before week 9.
11. **Which arguments get split when a splittable function takes several arrays?** For `matmul(a, b)` only `a` may be split (by rows), and `b` must stay whole. For a user function like `apply(images, masks)`, both must be split. Guessing by matching sizes fails for square matrices. → Each built-in op declares which arguments split (matmul: `a` only). For user functions, split every array argument along `batch_dim` and document that rule. Decide by week 6.
12. **Hardware.** This Mac has no NVIDIA GPU, so CUDA, NVML and `stress-ng` paths can only be checked on Colab or the fixed machine. → Name the fixed machine now and check whether anyone has an NVIDIA laptop for battery saver.

---

## Status after the week-8 code (2026-10-21, updated 2026-10-25)

✅ built and tested on a CPU-only machine (GPU code paths through a fake device) · 🟡 built, and its `gpu`-marked test waits for a real NVIDIA GPU (Colab) · ⏳ built, waits for data from the fixed machine · ❌ not done (team action, or week 9)

Tests: 107 pass on the dev Mac; 44 `gpu`-marked tests are skipped there and must pass on Colab (`pytest -m gpu`).

| Area | Status |
|------|--------|
| C-1 to C-5 | ✅ Python ≥ 3.11 (dev Mac 3.14), dependencies and extras in `pyproject.toml`, seeds fixed (inputs, model weights, forests, splits), CPU-only pytest with a fake GPU, 7 `shortcut:` comments |
| C-6 | ✅ `calibrate`, `features`, `predictor`, `monitor`, `history`, `bench/collect`, `bench/stress` hold no dispatch code. `bench/experiments.py` calls the public API, because measuring AutoHetero requires it. |
| API-1 to API-9 | ✅ `enable`, `auto`, `force`, `report`, `retrain` (Q1); tests in `test_dispatch`, `test_executor`, `test_trials`, `test_report`. API-6 also 🟡 (a real GPU model copy). |
| DSP-1 to DSP-4, DEV-1 to DEV-3 | ✅ including TST-2 and TST-7. Since step 24, a GPU failure during calibration (for example, GPU memory full) also falls back to the CPU instead of crashing the caller. DSP-2 also 🟡 (a real CUDA out-of-memory error). |
| OPS-1, OPS-3 | ✅ 10 ops, models in eval mode with no grad and fixed-seed weights |
| OPS-2 / TST-6 (weeks 1–2 exit) | 🟡 `tests/test_ops_parity.py`, 30 cases. A dry run on the Apple GPU passed all 30. |
| CAL-1 to CAL-5 | ✅ about 27 s on the dev Mac with the Apple GPU standing in; cached per machine key; skipped with no GPU. CAL-1 also 🟡 (real time on CUDA). |
| FEA-1, PRD-1 to PRD-8 | ✅ PRD-1 learns log(seconds per element), which is log(time) minus a known offset: beyond the largest size measured on a device, predictions grow instead of staying flat. PRD-6: user-function decisions come from fits and aren't cached, so they can't go stale. |
| EXE-1 to EXE-5 | ✅ TST-1, TST-5, trials, slowdown switch after 2 chunks (TST-1 also 🟡 on real CUDA). Since 2026-10-25, Hybrid rounds also take ≥ 5 ms (finding 1). |
| EXE-6 | 🟡/⏳ The constants sit together at the top of `executor.py`, with `HYBRID_MARGIN` in `predictor.py` and the sample period in `monitor.py`. All are untuned and wait for the fixed-machine data. |
| EXE-7 (week-6 exit) | 🟡 `tests/test_week6.py`. In the Apple-GPU dry run it switched after 2 chunks. |
| MON-1, MON-2 | ✅ CPU load from other processes only, median of about 1 s. Real NVML reading 🟡. |
| HIS-1 to HIS-3 | ✅ predicted next to actual for every call, online fits (saved to `fits.json` at exit and read back by the next session, since 2026-10-25), `retrain()`, log rotation at 50 MB. Retraining automatically when idle isn't built. |
| REP-1 | ✅ |
| BEN-1 to BEN-3 | ✅ code (`bench/collect.py`, `bench/stress.py`, `bench/experiments.py`) · ⏳ runs. `stress-ng` isn't on the dev Mac; the runner falls back to busy Python processes and records which tool it used. Since 2026-10-25, RQ2 skips a workload that doesn't fit in free GPU memory under its condition, as `collect.py` and Hybrid gain already did: on a small GPU, its forced GPU-only runs would fall back to the CPU and record CPU time as GPU-only. |
| BEN-4 | ⏳ Fixed machine named on 2026-10-25: a laptop with a Ryzen 7, an RTX 3050 Laptop GPU and 16 GB RAM, under WSL2. `RUNBOOK.md` is the step-by-step run on it. |
| BEN-5 (weeks 3–4 exit) | ⏳ `bench/collect.py`, then `eval/crossover.py` |
| EVL-1 to EVL-7 | ✅ scripts: `eval/rq1.py`, `rq2.py`, `rq3.py`, `overhead.py`, `hybrid_gain.py`, `crossover.py` · ⏳ numbers. "Replace every mock-up number in the deck" ❌ (team, after the runs). |
| DEM-1 (week-7 exit) | ✅ `demo/app.py`; `demo/pipeline.py` adds exactly one line (tested) |
| CO-1 | 🟡/⏳ `CARRYOVER.md` drafted on 2026-10-25: the §13 inferences mapped to the M.Tech review deck, with every measured cell pending the fixed-machine runs. |
| CO-2 to CO-4 | ❌ week 9, after submission. CO-3 needs the M.Tech run-log format (Q10). |
| CO-5 | ✅ All four references in DESIGN.md now carry full citations. Qilin, Grewe et al. (doi:10.1109/CGO.2013.6494993) and StarPU (CCPE 23(2), 2011, pp. 187–198, doi:10.1002/cpe.1631) were checked against their Crossref records on 2026-10-25. The ONNX Runtime page was checked live on 2026-10-21. |

**Open questions:** Q1, Q2, Q3, Q4, Q5, Q8, Q9 and Q11 are settled in code (see "Adopted" above). Q6 is settled as "train on calibration", but see finding 3 below. Q7 is settled (2026-10-25): see Q7 above. Q12 is settled (2026-10-25): see BEN-4. Q10 is still open.

**Findings from the dry runs (Apple GPU standing in), to settle on the fixed machine:**
1. **Hybrid rounds are costly on short calls.** Each 25% round waits for its GPU piece, so a matmul 2048 under GPU stress took 366 ms with AutoHetero against 22 ms on the CPU. Recommendation: give Hybrid rounds the same ≥ 5 ms minimum as chunks before the final runs. *Done 2026-10-25: a round is at least 5 ms of predicted work with both devices busy, so a short call runs in one round. The GPU piece is still slow under stress, because the forest doesn't know about load (finding 3).*
2. **False slowdown switches at idle.** The forest predicts in steps between the ~6 calibrated sizes, so a chunk can look 1.8–1.9× slow on an idle machine. Tune `SLOWDOWN` on the real data; how often switches fire is a §13 measurement anyway.
3. **Built-in op decisions ignore load.** The forest trains on idle calibration, so its load features are constant. Under sustained contention, short built-in calls stay on a slowed GPU; only long splittable calls are rescued, by the per-call switch. User functions adapt through their fits. Options: let built-ins use their fits too once measured, or calibrate under load as well.
4. Per-function fits live only in memory, so every new session repeats each user function's trial. *Done 2026-10-25: fits are saved to `fits.json` in the machine cache at exit, so a trial runs once per machine. `bench/experiments.py` deletes the file at the start of each run so the trial still shows up in rep 0.*

**End-to-end rehearsal (2026-10-21, dev Mac, Apple GPU standing in):** the whole runbook ran: calibration (20.3 s), the dataset (1155 runs), RQ2 under idle, GPU stress and CPU stress (378 runs), Hybrid gain (384 runs), and the overhead measurement. Every `eval/` script produced its figures and tables (12 files). The rehearsal found three problems in the results, now fixed and tested:
- The RQ3 "decision miss" row wasn't a forest miss. It now times a real forest decision (4.4 ms), and the cached built-in decision is reported separately.
- Hybrid gain counted runs that never split. A batch of 1 can't split, and a speed-based split can give one device nothing. Each run now records `gpu_share`, and only runs where both devices worked count as Hybrid.
- (cosmetic) Crowded axis labels on the Hybrid-gain figure.

No rehearsal number belongs in the report: they come from the Apple GPU, not the fixed NVIDIA machine.
