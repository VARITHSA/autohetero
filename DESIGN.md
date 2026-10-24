# AutoHetero — Design

**AutoHetero: A Self-Adaptive, Zero-Configuration Runtime Layer for CPU–GPU Workload Dispatch in Python Applications**

Research Methodology and Technical Communication · team of 8.

AutoHetero decides, **for every call**, whether a Python workload runs on the CPU, the GPU, or both (Hybrid). It benchmarks the user's machine on first run, predicts the fastest option, runs big tasks in chunks, and rebalances onto both devices if a task runs slower than predicted. Measured times feed back so decisions improve on each machine.

```
predict → execute in chunks → monitor → rebalance (CPU / GPU / Hybrid) → learn
```

The concept document and the deck define *what* we claim; this file defines *how* we build it.

---

## 1. What we must answer

| RQ | Question | Evidence |
|----|----------|----------|
| RQ1 | Can a lightweight learned model predict whether CPU or GPU is faster for a workload and input size? | Prediction accuracy on held-out runs (target ≥ 90%) |
| RQ2 | Can runtime monitoring catch slowdowns early enough to recover time by rebalancing? | Total time under induced contention vs. GPU-only |
| RQ3 | Can this work with zero configuration and one-line integration, without slowing the user's code? | Dispatch overhead (target < 1 ms/call), lines changed in the demo app |

Every target in the concept doc and deck (≥ 90%, within 5%, 10–30% Hybrid gain, < 1 ms) is a **hypothesis**. Never hardcode, assert or test against them. Report what is measured, including targets that are missed.

"Within 5%" belongs to RQ2: AutoHetero's median time on a workload is at most 5% above the oracle's (the best measured option in each run). `eval/rq2.py` reports the share of workloads that meet it, per condition.

## 2. Public API (keep exactly this surface)

```python
import autohetero
autohetero.enable()             # Level 1: dispatch every @auto function and built-in op
@autohetero.auto                # Level 2: one function
def enhance(images): ...
with autohetero.force("cpu"):   # Level 3: manual override ("cpu" | "gpu" | "hybrid")
    enhance(images)
autohetero.report()             # what ran where, time taken, estimated time saved
autohetero.retrain()            # retrain the predictor on this machine's measured calls (never inside a call)
```

- `@auto(splittable=True, batch_dim=0)` marks a function whose output along the batch dimension depends only on the matching input slice. Only splittable functions get chunking and Hybrid mode. **Default is `splittable=False`**: splitting `sort` or `groupby` silently gives wrong answers, and correctness beats speed.
- `@auto` functions must **not modify their inputs in place** (`x += 1`, `x.clamp_()`). On the CPU, inputs share memory with the caller's arrays; on the GPU they are copies. In-place edits would change the caller's data only on some devices, and would corrupt the user-function trial (§6), which runs on both devices — so the trial runs its GPU chunk first.
- Level 1 does **not** monkey-patch PyTorch globally. It enables dispatch for `@auto` functions and the built-in `autohetero.ops`. Patching all of torch is out of scope.
- Inputs may be `torch.Tensor` or `numpy.ndarray`. Outputs come back in the same type and on the same device as the inputs. The user never writes `.to("cuda")`.
- **Models:** `nn.Module` arguments are moved too. `devices.py` keeps one cached copy per device (weights copied once, not per call), which is what lets ResNet-18 / MobileNetV2 run in Hybrid mode. A model captured in a closure or global can't be moved — `@auto` functions must take it as an argument; built-in model ops manage their own copies.
- **Nesting:** the outermost dispatched call decides. Built-in ops called inside an `@auto` function run on the device already chosen, so data isn't shuttled back and forth mid-function.
- **Decision log:** one line per new decision and per mode switch (`[AutoHetero] enhance: batch 200, GPU`, `[AutoHetero] Slowdown 1.8x, Hybrid`), plus calibration start/finish. `AUTOHETERO_QUIET=1` silences it.

## 3. Scope

- **In:** one machine, one NVIDIA GPU + CPU, PyTorch 2.x workloads, inference and data processing.
- **Out:** training workloads, multi-GPU, NPUs, distributed systems, patching third-party libraries.
- **Known limits:** Hybrid only for splittable batches; very small inputs aren't chunked; a single launched kernel can't be rebalanced.

## 4. Repository layout

```
autohetero/
├── DESIGN.md
├── pyproject.toml               # pip install -e . ; later published as `autohetero`
├── autohetero/
│   ├── __init__.py              # enable, auto, force, report
│   ├── dispatch.py              # the @auto wrapper: features → decide → run → return
│   ├── devices.py               # move inputs/outputs, numpy↔torch, CPU fallback
│   ├── calibrate.py             # first-run machine benchmark (~30 s), cached
│   ├── features.py              # feature vector for a call
│   ├── predictor.py             # per-device time model + decision cache
│   ├── executor.py              # chunked execution, slowdown check, Hybrid split
│   ├── monitor.py               # background CPU/GPU utilisation sampler
│   ├── history.py               # measured-call log, online updates
│   ├── report.py
│   └── ops/                     # built-in ops: blur, resize, sobel, resnet18, mobilenetv2,
│                                #   filter, sort, groupby, matmul, fft
├── bench/                       # benchmark suite, dataset collection, stress scenarios
├── eval/                        # scripts producing every graph and table in the report
├── demo/app.py                  # Streamlit demo
└── tests/
```

## 5. How a call is handled (`dispatch.py`)

1. **Force check:** inside `force(...)`, skip prediction (a `contextvars` variable, so it's thread-safe).
2. **Features** (`features.py`): op type (built-in name, or `"user"`), input elements, input bytes, batch size, dtype, current CPU and GPU utilisation from the monitor's cached sample.
3. **Decide** (`predictor.py`): predicted time per mode; pick the fastest.
4. **Execute** (`executor.py`): direct for non-splittable or small inputs, otherwise chunked (§7).
5. **Return** results in the caller's type and device.
6. **Record** the measured time in `history.py`.

**Safe fallback:** any GPU error (no CUDA, OOM, driver error) → rerun on CPU, mark GPU unhealthy for that function and size bucket for the session, log once. AutoHetero must never crash user code. Document that a function with side effects may run twice on fallback.

## 6. Prediction (`predictor.py`, `calibrate.py`)

**Calibration (first run, ~30 s, `calibrate.py`):**
- Microbenchmarks: host↔device bandwidth, kernel launch latency, and each built-in op on CPU and GPU at ~6 log-spaced sizes.
- Saved to `~/.cache/autohetero/<machine-key>/` where machine-key = CPU model + GPU name + driver + torch version. Recalibrate when the key changes.
- No GPU → skip calibration and use CPU always.
- **Time budget ~30 s** (the deck's stated target): cap each microbenchmark by time, not size — e.g. ResNet-18 on CPU at batch 256 alone can take several seconds, so measure the large CPU sizes at fewer points and let the model interpolate. Print progress.

**Model:**
- Random forest regressor (scikit-learn) on `log(time)`, one per device, trained on the calibration + benchmark dataset. Gives predicted times, so it powers both the decision and the 1.5× slowdown check. A decision tree is trained alongside for the RQ1 comparison.
- **User functions** (op type `"user"`) have no calibration data. The first call runs its first chunk on both devices (a *trial*). After that, a per-function log-log fit of time vs. input size per device, updated online, predicts later calls.
- **Overhead trap:** `RandomForest.predict` on one sample takes milliseconds — too slow for the < 1 ms target. Cache decisions keyed by `(op, log2 size bucket, dtype, load bucket)`; call the forest only on a cache miss. Invalidate a cache entry when online updates move its prediction.

**Hybrid split:** `gpu_share = s_gpu / (s_gpu + s_cpu)` from measured or predicted speeds (6× faster GPU → ~85% of the work), so both devices finish together.

**Hybrid as an up-front choice:** for splittable calls, Hybrid competes with CPU and GPU from the start (the deck's demo picks it for Classify on an idle machine). Predicted Hybrid time = `T_cpu·T_gpu / (T_cpu + T_gpu) + split_overhead`, with `split_overhead` (thread start, chunk copy, concatenation) measured during calibration. If Hybrid doesn't beat the faster single device by a margin, don't pick it.

## 7. Chunked execution and adaptation (`executor.py`)

For splittable inputs above a minimum size:
1. Split along `batch_dim`. First chunk ≈ 10% of the batch, but large enough to take ≥ ~5 ms (smaller chunks are dominated by launch overhead).
2. After each chunk, compare actual time with the predicted time for that chunk.
3. If **actual > 1.5 × predicted**, switch to Hybrid for the remaining chunks, re-splitting by the measured speeds. Keep re-checking and re-splitting after each round.
4. Hybrid runs two threads, one feeding the GPU and one running CPU ops. PyTorch releases the GIL inside ops, so both run concurrently.
5. Concatenate results in the original order.

Constants (1.5×, 10%, 5 ms) live in one place and are tuned on benchmark data, not guessed per call site.

## 8. Monitoring and learning (`monitor.py`, `history.py`)

- A background daemon thread samples CPU utilisation (`psutil`) and GPU utilisation (NVML via `nvidia-ml-py`) every ~200 ms. Calls read the cached value; no syscalls on the hot path.
- Every measured call is appended to the history. Per-function fits update immediately. Retraining the forest from history is done at idle or on `autohetero.retrain()`, never inside a user call.

## 9. Report (`report.py`)

`report()` prints a table: function, mode chosen, calls, total time, **estimated** time saved vs. always-GPU (from predicted times, labelled as an estimate), and a count of fallbacks and Hybrid switches.

## 10. Benchmarks and evaluation (`bench/`, `eval/`)

**Workloads** (all PyTorch, one code path on either device):

| Domain | Workloads | Sizes |
|--------|-----------|-------|
| Image and video | Gaussian blur, resize, Sobel edge detection | small → 4K images, batches 1–256 |
| AI/ML inference | ResNet-18, MobileNetV2 | batch 1–256 |
| Data analytics | filter, sort, group-by aggregation | 10³–10⁷ rows |
| Scientific and signal | matrix multiplication, FFT | log-spaced to GPU memory limit |

**Data collection:** every workload × size × {CPU, GPU}, 5 repetitions, 1 warm-up discarded, time **including transfers**. `torch.cuda.synchronize()` before stopping any GPU timer.

**Model validation (RQ1):** 80/20 split and k-fold cross-validation, as in the concept doc. Also report leave-one-op-out accuracy, since that's how the model meets unseen user functions.

**Conditions (RQ2):** idle; GPU stress (a parallel PyTorch GPU job); CPU stress (`stress-ng`); battery saver. Battery saver needs a laptop with an NVIDIA GPU — Colab and desktops can't test it. If no team member has one, drop the condition and say so in the report.

**Hybrid gain experiment:** on the idle machine, Hybrid vs. GPU-only across batch sizes for every splittable workload (the deck's 10–30% hypothesis). Expect no gain on small batches; show where it starts.

**Demo (Streamlit):** the CPU-only and GPU-only columns come from rerunning the same steps under `force("cpu")` and `force("gpu")`.

**Baselines:** always-CPU, always-GPU, and an oracle (best measured option per run). The gap to the oracle is the honest measure of decision quality.

**Metrics:** mean speedup, prediction accuracy, recovery time after a slowdown, dispatch overhead per call, variance across runs.

**Rules:**
- Final numbers come from one fixed machine. Use Colab for development only — its GPUs and load vary between sessions.
- AutoHetero's overhead (features, prediction, trials, chunking) is inside every AutoHetero number.
- Every graph and table comes from a script in `eval/` reading the collected data. No hand-edited numbers.

## 11. Implementation phases

Mapped to the 8-week timeline. A phase is done only when its exit check passes.

| Weeks | Build | Exit check |
|-------|-------|------------|
| 1–2 | Environment, `ops/` workloads, `devices.py` | Every op gives matching CPU and GPU results (within float tolerance) at 3 sizes |
| 3–4 | `bench/` suite and dataset, `calibrate.py`, `monitor.py` | Dataset collected; crossover graph per op shows where CPU beats GPU |
| 5 | `features.py`, `predictor.py` with cache, `dispatch.py`, `force`, fallback | RQ1 accuracy reported; overhead measured per call; GPU-less machine runs everything on CPU without errors |
| 6 | `executor.py` chunking, slowdown check, Hybrid, `history.py` online updates, user-function trials | Under GPU stress, a long task switches to Hybrid within 1–2 chunks; Hybrid results identical to single-device results |
| 7 | Controlled experiments, `report.py`, Streamlit demo | All conditions × baselines measured; demo needs one added line |
| 8 | `eval/` graphs and tables, report, presentation | Every claim in the report backed by a generated figure; replace every mock-up number in the deck |
| 9 (after submission) | Carry-over to the M.Tech project (§13) | Carry-over note written; reusable modules tagged; M.Tech motivation backed by AutoHetero's measured results |

## 12. Conventions

- Python ≥ 3.11 (`requires-python` in `pyproject.toml`; the dev Mac runs 3.14), PyTorch 2.x, NumPy, scikit-learn, psutil, nvidia-ml-py, matplotlib, Streamlit (demo only). Nothing else without a stated reason.
- torchvision: the standard ResNet-18 / MobileNetV2 definitions for the built-in model ops, used with random weights and a fixed seed (no weight downloads).
- Seeds fixed everywhere (data, forests, splits).
- `pytest` must run on a CPU-only machine: GPU paths are tested via a fake device; real-GPU tests are marked `@pytest.mark.gpu`.
- Must-have tests: Hybrid output equals single-device output; fallback when CUDA raises; `force()` respected across threads; numpy in → numpy out; non-splittable functions are never chunked.
- Shortcuts get a comment: `shortcut: <the limit>, <when to upgrade>`.

## 13. Phase 9 — Carry-over to the M.Tech project

The M.Tech project, *Intelligent Execution Strategy Selection*, extends AutoHetero from **per-call** decisions inside one program to **per-job** decisions on a shared server: CPU or GPU, plus how many GPU jobs run at once (NVIDIA MPS). It adds three things AutoHetero doesn't have: a confidence score on every prediction, measuring only when confidence is low, and the GPU-concurrency decision.

**Rule:** AutoHetero is cited in the M.Tech work as prior work by the 8-member team, with the team's agreement and the guide informed. AutoHetero's results support only what AutoHetero measured; every M.Tech claim needs new M.Tech measurements.

**Prepare during weeks 1–8 (cheap now, costly later):**
- `history.py` logs the **predicted** times next to the actual times for every call — the M.Tech confidence work is calibrated against exactly this gap.
- Keep the random forest's per-tree predictions reachable (`estimators_`); don't wrap them away.
- Keep `calibrate.py`, `features.py`, `predictor.py`, `monitor.py` and `bench/` free of decorator/dispatch code, so they lift out as a library.

**Inferences to draw, each from a specific AutoHetero result:**

| AutoHetero result | Inference for the M.Tech project |
|---|---|
| Leave-one-op-out accuracy (RQ1) | How badly the model does on unseen workloads → the case for a confidence score instead of trusting every prediction |
| How often user-function trials and 1.5× slowdown switches fired, and what they cost | What measuring costs and how often it was actually needed → the case for measuring *only* when unsure |
| Hybrid gain vs. batch size | If the gain is small or narrow, it justifies the M.Tech project dropping Hybrid and focusing on concurrency |
| Results under GPU stress | How much one GPU job slows another → the motivation for choosing GPU concurrency deliberately rather than maximising it |
| Crossover graphs per op | Evidence that the right device depends on input size, used in the M.Tech problem statement |
| Dispatch overhead | The budget for the M.Tech decision path |

**Deliverables:**
1. `CARRYOVER.md` — what was reused, what changed, and the table above filled with AutoHetero's measured numbers (no targets, no mock-up values).
2. A git tag (`autohetero-v1`) on the code the M.Tech project starts from.
3. The benchmark dataset exported in the format the M.Tech run log expects, clearly marked as AutoHetero data — usable for motivation and sanity checks, never as M.Tech evaluation data.

## References

1. C.-K. Luk, S. Hong, H. Kim, "Qilin: Exploiting Parallelism on Heterogeneous Multiprocessors with Adaptive Mapping," Proc. MICRO-42, 2009, pp. 45–55, doi:10.1145/1669112.1669121.
2. D. Grewe, Z. Wang, M. O'Boyle, "Portable Mapping of Data Parallel Programs to OpenCL for Heterogeneous Systems," Proc. CGO, IEEE, 2013, pp. 1–10, doi:10.1109/CGO.2013.6494993.
3. C. Augonnet, S. Thibault, R. Namyst, P.-A. Wacrenier, "StarPU: A Unified Platform for Task Scheduling on Heterogeneous Multicore Architectures," Concurrency and Computation: Practice and Experience, vol. 23, no. 2, 2011, pp. 187–198, doi:10.1002/cpe.1631. (Not the Euro-Par 2009 paper with the same title.)
4. ONNX Runtime documentation, "Execution Providers," https://onnxruntime.ai/docs/execution-providers/.

1–3 were checked against their Crossref records on 2026-10-25, and 4 on the live page on 2026-10-21.
