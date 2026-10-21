# AutoHetero

AutoHetero picks where your PyTorch code should run — CPU, GPU, or both at once — for every single call. You add one line, and it handles the rest.

I built it for a course project. The idea came from something I kept running into: the GPU isn't always faster. For small inputs, copying the data over and starting the GPU costs more than just doing the work on the CPU. For big inputs the GPU wins easily. Where that switch happens depends on the operation, the size of the input, your machine, and whatever else is running on it right now. Guessing `"cpu"` or `"cuda"` by hand every time gets old fast, and you'll often guess wrong.

So AutoHetero measures your machine once, predicts the fastest option for each call, and keeps an eye on things while the call runs. If the GPU turns out slower than expected (say another program grabbed it), it moves part of the work to the CPU so both finish together. Every call gets timed, and it learns from that.

> It's a research prototype (version 0.1.0). It's meant for one machine with one NVIDIA GPU, running PyTorch 2.x inference and data processing — not training. Read [Limitations](#limitations) before you rely on it.

---

## Contents

- [What hardware it works on](#what-hardware-it-works-on)
- [Installing it](#installing-it)
- [Quick start](#quick-start)
- [How it decides](#how-it-decides)
- [API reference](#api-reference)
- [Built-in ops](#built-in-ops)
- [Writing your own functions](#writing-your-own-functions)
- [The log](#the-log)
- [Settings and files](#settings-and-files)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Running the tests](#running-the-tests)
- [Reproducing my results](#reproducing-my-results)
- [The demo](#the-demo)
- [What's where](#whats-where)
- [Contributing](#contributing)
- [References](#references)

---

## What hardware it works on

| Your machine | What you get |
|--------------|--------------|
| NVIDIA GPU on Linux or Google Colab, with a CUDA build of PyTorch | Everything: CPU, GPU and Hybrid. This is what I built it for. It only uses GPU 0. |
| NVIDIA GPU on Windows | Should work. I haven't tested it yet. |
| AMD GPU with a ROCm build of PyTorch | Probably works, because ROCm builds pretend to be `torch.cuda`. It can't read the GPU's load, though. Not tested. |
| Apple Silicon, Intel GPU, or no GPU at all | It still runs, just with everything on the CPU. It tells you so once. |

You need Python 3.11 or newer, plus PyTorch 2.x, torchvision, NumPy, scikit-learn, psutil and nvidia-ml-py 12 or newer. Matplotlib (for the figures) and Streamlit (for the demo) are optional.

---

## Installing it

```bash
git clone <this repository>
cd autohetero
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # just the library
pip install -e ".[dev]"            # plus pytest and matplotlib, for the tests and figures
pip install -e ".[demo]"           # plus Streamlit, for the demo
```

If you're on a GPU machine, do this check before anything else. It should print `True` and the name of your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

Got `False`? Then your PyTorch can't see the GPU. Usually that means it's a CPU-only build (on Windows, plain `pip install torch` gives you one) or your NVIDIA driver is too old. Grab the CUDA build that matches your driver from [pytorch.org](https://pytorch.org/get-started/locally/) and check again. AutoHetero will still work without it, but everything will run on the CPU.

---

## Quick start

There are three ways to use it. Pick whichever fits.

### 1. One line, for the built-in ops

```python
import torch
import autohetero

autohetero.enable()                      # the one line: built-in ops now run where they're fastest

images = torch.rand(16, 3, 720, 1280)    # on the CPU, as usual
edges = autohetero.ops.sobel(autohetero.ops.blur(images))   # each op is dispatched; the result comes back on the CPU
```

The very first call on a new machine runs a calibration. It takes about 30 seconds, prints its progress, and only happens once. After that, every new input size gets one decision, and you'll see a line like this (which device it picks depends on your machine):

```
[AutoHetero] blur: batch 16, GPU
[AutoHetero] sobel: batch 16, HYBRID
```

### 2. Your own functions, with `@autohetero.auto`

```python
import numpy as np
import torch
import autohetero

@autohetero.auto(splittable=True)
def normalize(x):
    # x arrives as a torch tensor on whichever device AutoHetero chose
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)

out = normalize(np.random.rand(10_000, 512).astype(np.float32))   # numpy in -> numpy out
```

Models work too. Pass the model in as an argument and AutoHetero keeps one copy of it on each device:

```python
import torchvision

model = torchvision.models.resnet18(weights=None).eval()

@autohetero.auto(splittable=True)
def classify(images, model):
    with torch.no_grad():
        return model(images).argmax(1)

batch = torch.rand(64, 3, 224, 224)
labels = classify(batch, model)
```

### 3. Taking over by hand, with `autohetero.force`

```python
with autohetero.force("cpu"):      # or "gpu", or "hybrid"
    labels = classify(batch, model)
```

### Seeing what happened

```python
autohetero.report()
```

You'll get something like this (your numbers and devices will be different):

```
function  calls  modes                time  saved vs GPU (est.)  fallbacks  switches
--------  -----  ---------------  --------  -------------------  ---------  --------
blur          3  GPU 3              91.2 ms             12.0 ms          0         0
classify      2  HYBRID 1, GPU 1   612.0 ms             48.0 ms          0         1
--------  -----  ---------------  --------  -------------------  ---------  --------
total         5                    703.2 ms             60.0 ms          0         1
saved vs GPU is an ESTIMATE: predicted always-GPU time minus measured time, summed over calls that had a prediction ...
```

---

## How it decides

Here's what goes on under the hood, roughly in order.

**Calibration (first use, about 30 seconds).** It times a handful of things on your machine: how fast data copies to and from the GPU, how long it takes to start work on the GPU, the extra cost of running on both devices at once, and every built-in op on the CPU and GPU at up to six sizes. The 30 seconds are shared between the ops. Big sizes that would take too long on the CPU get skipped — it predicts each one from the sizes it has already timed. All of this gets saved in `~/.cache/autohetero/`, under a key made from your CPU, GPU, NVIDIA driver and PyTorch version. Change any of those and it calibrates again by itself.

**Predicting.** Each call gets described by a few numbers: which op it is (or "your function"), how many elements and bytes its arrays hold, the batch size, the dtype, and how busy the CPU and GPU are right now.

- For the built-in ops, two random forests (one per device) predict the time. They learn time *per element*. I did that on purpose: plain forests predict a flat line past the biggest size they've seen, which would make huge inputs look cheap on the CPU. A forest prediction takes a few milliseconds, which is way too slow to do on every call. So each decision gets cached by function, size (rounded to a power of 2), dtype and load. After the first call at a size, deciding takes microseconds.
- Your own functions are different. AutoHetero has never seen them, so it learns from their own timings — a simple fit of time against input size for each device, updated after every call, with old measurements slowly fading out.
  - The first call of a splittable function is a **trial**. About 10% of the batch runs on the GPU, then another 10% on the CPU, and the rest gets split by the speeds it just measured.
  - A function that isn't splittable can't do that. Its first call goes with the forests' best guess, and its second call runs once on the other device so both get measured.
- Hybrid is only an option for splittable calls with at least 2 items. Its predicted time is `T_cpu·T_gpu/(T_cpu+T_gpu)` plus the cost of the extra thread and of joining the results. And it has to beat the faster single device by at least 10%, or it doesn't get picked.

**Running.** Short calls, and calls that can't be split, just run in one go. Long splittable calls run in chunks of about 10% of the batch, each taking at least 5 ms. From the second chunk on, each chunk's time is checked against its own prediction. (The first chunk gets a pass, since it also pays one-time costs like copying a model to the GPU.) If a chunk takes more than 1.5× as long as predicted, the rest of the call switches to **Hybrid**. That means rounds of 25% of the batch, split between the GPU (on a worker thread) and the CPU (on your thread), sized so both finish at the same time. After each round it re-measures and re-splits. At the end, the pieces get stitched back together in order, as the same type and on the same device your input came from.

**Learning.** Every call goes into a history log, with the predicted time right next to the measured one. Your functions' fits update straight away. The forests only learn from that history when you call `autohetero.retrain()`.

**When something breaks.** If the GPU throws an error during a call (out of memory, a CUDA or driver error), the call just runs again on the CPU. It logs that once, and that function at that size stays on the CPU for the rest of the session. If the CPU run fails too, the bug is in your function, and you get your normal error. And if the GPU fails during calibration, AutoHetero stops using it for the session instead of crashing your program.

---

## API reference

Everything lives in the `autohetero` package.

### `autohetero.enable()`

Turns on dispatch for the [built-in ops](#built-in-ops), starts the background load monitor, and calibrates the machine if it hasn't been done yet. Call it once when your program starts. You don't need it for `@auto` functions — those are always dispatched.

### `@autohetero.auto` or `@autohetero.auto(splittable=False, batch_dim=0)`

Put this on a function and every call to it runs wherever it's predicted to be fastest.

- `splittable=True` is a promise: row *i* of the output depends only on row *i* of the inputs (along `batch_dim`). Only splittable calls can be chunked or split across both devices. It's `False` by default, and for good reason — splitting something like a sort or a group-by gives you wrong answers.
- `batch_dim` is the dimension inputs get split along and outputs get joined along. It defaults to `0`.

A few more things worth knowing:

- Arguments can be torch tensors, numpy arrays, or lists, tuples and dicts of them, mixed with anything else. Every array gets moved to the chosen device, and numpy arrays turn into tensors. So **your function always receives torch tensors**.
- The result comes back as the same type, on the same device, as the **first** array you passed in. Numpy in gives numpy out on the CPU. A CUDA tensor in gives a CUDA tensor out.
- `nn.Module` arguments get moved too, using one cached copy per device. Your own model object is never touched.
- In a splittable call, **every** array argument gets split along `batch_dim`. If something has to stay whole, keep it inside a model or a non-array argument, or don't mark the function splittable.
- A call with no array arguments just runs normally.
- If a dispatched call happens inside another one, it runs on whatever device the outer call picked.

### `autohetero.force(mode)`

A `with` block. Inside it, dispatched calls skip the prediction and run on `mode`:

- `"cpu"` — everything on the CPU, in one piece.
- `"gpu"` — everything on the GPU, in one piece.
- `"hybrid"` — splittable calls with 2 or more items get split across both devices. Anything else runs on whichever single device is faster.

Anything else raises a `ValueError`. `force` only affects the current thread (or async task). Threads you start inside the block won't pick it up. On a machine with no GPU, every mode runs on the CPU.

### `autohetero.report()`

Prints one row per function for this session: how many calls, which modes it picked, total time, roughly how much time it saved compared to always using the GPU, how many times it fell back to the CPU after a GPU error, and how many times it switched to Hybrid. It also returns the rows as a list of dicts. Keep in mind the "saved" column is an **estimate** (predicted GPU-only time minus measured time). It only counts calls that actually had a prediction.

### `autohetero.retrain()`

Retrains the forests on your machine's calibration plus the built-in op calls in its history (up to the newest 50,000), then clears the decision cache. It never runs in the middle of a call. A good time to use it is after a big batch job, so the next run benefits from what got measured.

### `autohetero.ops`

The [built-in ops](#built-in-ops). `autohetero.ops.RAW` holds the plain versions that never get dispatched.

---

## Built-in ops

These are plain PyTorch. They're what `enable()` dispatches, what calibration times, and what my benchmarks use. Without `enable()` (and outside an `@auto` function) they just run on whatever device their input is already on.

| Op | Input | Output | Can be split? |
|----|-------|--------|---------------|
| `blur(images, sigma=2.0)` | float32/float64 tensor shaped `(..., H, W)`, e.g. `(N, C, H, W)` | Gaussian blur, same shape | yes |
| `resize(images, scale=0.5)` | same as above | bilinear, antialiased resize by `scale` | yes |
| `sobel(images)` | same as above | edge strength, same shape | yes |
| `resnet18(images)` | float32 `(N, 3, H, W)`, usually 224×224 | ImageNet logits `(N, 1000)` | yes |
| `mobilenetv2(images)` | same as above | ImageNet logits `(N, 1000)` | yes |
| `filter(table, threshold=0.5)` | float tensor `(rows, columns)` | rows whose first column is above `threshold`, in order | yes |
| `sort(table)` | same as above | rows sorted by their first column (stable) | **no** |
| `groupby(keys, values)` | integer `keys (N,)`, `values (N, ...)` | `(distinct keys, sum of values per key)` | **no** |
| `matmul(a, b)` | two matrices | `a @ b` | yes (rows of `a`; `b` stays whole) |
| `fft(signals)` | `(signals, samples)` | complex FFT along the last dim | yes (across signals) |

One heads-up about the two models: they use torchvision's definitions with **random weights from a fixed seed**, so nothing gets downloaded. They're real workloads for timing, but they won't classify anything correctly. Also, the image ops want float32 or float64, so convert `uint8` images first.

---

## Writing your own functions

Stick to these and AutoHetero can move your function around freely.

1. **Use torch operations.** Your inputs show up as tensors on the chosen device, so write `x.mean(dim=1)`, not `np.mean(x, axis=1)`.
2. **Don't change your inputs in place** (`x += 1`, `x.clamp_()`). On the CPU, the inputs share memory with your original arrays; on the GPU they're copies. An in-place edit would change your data on some runs and not others.
3. **Pass models in as arguments.** A model stuck in a closure or a global variable can't be moved to the GPU.
4. **Side effects can happen twice.** It's rare, but if the GPU fails partway through, the whole call reruns on the CPU.
5. **Only say `splittable=True` when row *i* of the output depends only on row *i* of the inputs.** Per-sample models, element-wise maths, per-image filters — fine. Sorting, normalising across the whole batch, group-bys, anything that mixes rows — not fine.
6. **Return arrays from splittable functions** (or tuples, lists or dicts of arrays), so the pieces can be glued back together.

If your data is laid out as `(features, batch)`, use `@autohetero.auto(splittable=True, batch_dim=1)`.

---

## The log

AutoHetero writes short lines to **stderr** so you can see what it's doing:

- `Calibrating this machine once (~30 s): <CPU> + <GPU>`, then a line per op, then `Calibration done in 27.4 s, cached in …` — first use on this machine.
- `enhance: batch 64, GPU` — a new decision, or a changed one, for this function at this size.
- `enhance: batch 64, TRIAL` — the first call of a splittable function, measuring both devices.
- `enhance: slowdown 2.3x, Hybrid` — a chunk ran 2.3× slower than expected, so the rest got split across both devices.
- `enhance: GPU failed (OutOfMemoryError: …); this size runs on the CPU from now on` — the call was rerun on the CPU.
- `no usable CUDA GPU: everything runs on the CPU (…)` — shown once when there's no GPU to use, with a hint if PyTorch can't see CUDA.
- `GPU unusable (…); everything runs on the CPU this session` — the GPU failed during calibration.

Set `AUTOHETERO_QUIET=1` if you'd rather not see any of it.

---

## Settings and files

There's nothing you have to configure. The few knobs that do exist:

- **Silence the log:** set the environment variable `AUTOHETERO_QUIET=1`.
- **Recalibrate from scratch:** delete `~/.cache/autohetero/`. (It also recalibrates on its own when your CPU, GPU, driver or PyTorch version changes.)
- **Put the cache somewhere else:** set `autohetero.calibrate.CACHE = Path(...)` before your first call.

What ends up in `~/.cache/autohetero/<machine key>/`:

- `calibration.json` — your machine, copy speeds, launch time, Hybrid costs, and how long calibration took.
- `timings.csv` — the op timings from calibration (same format as my benchmark data).
- `history.jsonl` — one line per call: what it looked like, which mode ran, the predicted times, the measured time, where each piece ran, and any fallback or slowdown. Once it hits 50 MB it gets renamed to `history.old.jsonl`, so the two never use more than about 100 MB.

If you want to experiment, these are the tuning constants. They're my first guesses, and I plan to tune them on real benchmark data:

| Constant | Default | What it does |
|----------|---------|--------------|
| `executor.SLOWDOWN` | 1.5 | how much slower than predicted a chunk can be before switching to Hybrid |
| `executor.FIRST_CHUNK` | 0.1 | share of the batch in each chunk |
| `executor.MIN_CHUNK_SECONDS` | 0.005 | shortest a chunk is allowed to be |
| `executor.HYBRID_ROUND` | 0.25 | share of the batch in each Hybrid round |
| `predictor.HYBRID_MARGIN` | 0.1 | how much Hybrid has to beat the best single device by |
| `calibrate.BUDGET` | 30.0 | seconds spent calibrating the ops |
| `history.DECAY` | 0.95 | how fast old timings fade out of a function's fit |
| `monitor.PERIOD` | 0.2 | seconds between load readings |

---

## Troubleshooting

**Everything runs on the CPU, but I have a GPU.** Check the log for `no usable CUDA GPU`, then run the GPU check from [Installing it](#installing-it). Nine times out of ten it's a CPU-only PyTorch build or an old driver.

**The first call takes forever.** That's calibration — about 30 seconds, once per machine, and it's cached afterwards. Only delete `~/.cache/autohetero/` if you actually want to redo it.

**CPU, GPU and Hybrid give slightly different numbers.** That's normal. Each device does the floating-point maths in a different order, so results can differ in the last few digits (about 1e-6 relative). Newer NVIDIA GPUs (Ampere and up) also use TF32 for convolutions by default, which can push the difference to around 1e-3. If you need them closer, set `torch.backends.cudnn.allow_tf32 = False`.

**It picked the "wrong" device.** It's making predictions, and predictions miss sometimes. Wrap the call in `autohetero.force(...)` if you know better. Run `autohetero.retrain()` after a workload so the forests learn from what actually happened. And have a look at `history.jsonl` to compare predicted and measured times.

**`force("gpu")` is ignored inside a thread I started.** That's on purpose — `force` is per thread. Put the `with` block inside the thread.

**My function works on its own but breaks under `@auto`.** It's usually one of three things: numpy calls on what's now a torch tensor, `splittable=True` on something that mixes rows, or a return value that can't be stitched back together. See [Writing your own functions](#writing-your-own-functions).

**GPU out of memory.** AutoHetero reruns the call on the CPU and keeps that function at that size on the CPU for the rest of the session. Smaller batches help, and so does `force("hybrid")` for splittable functions, since it puts less on the GPU.

---

## Limitations

I'd rather you hear these from me.

- **Scope.** One machine, one NVIDIA GPU (GPU 0), PyTorch 2.x inference and data processing. No training, no multiple GPUs, nothing distributed. If you choose a GPU with `CUDA_VISIBLE_DEVICES`, the load reading might come from the wrong GPU.
- **Hybrid needs a splittable call with 2 or more items.** Very short calls aren't chunked, and once a single GPU kernel is launched it can't be rebalanced.
- **The built-in ops don't react to long-running load yet.** The forests learn from calibration on an idle machine, so the load numbers don't tell them anything. A long splittable call still gets rescued by the chunk check, but short calls on a busy GPU stay stuck there. Your own functions do adapt, through their fits.
- **Hybrid can cost more than it saves on short calls.** Each round has to wait for its GPU piece. I'm planning to tune this (see REQUIREMENTS.md).
- **Function fits only live in memory.** Start a new Python session and every user function does its trial again.
- **Model copies are cached per device.** If you change a model's weights after its first dispatched call, the GPU copy still has the old ones.
- **There's a small overhead.** On my development machine it was about 10 µs per call once a size had been decided, and a few milliseconds the first time a built-in op meets a new size. So don't bother decorating tiny calls that run in a tight loop.
- **The demo models have random weights.**

---

## Running the tests

```bash
pip install -e ".[dev,demo]"
pytest                      # everything that runs without a GPU (the GPU tests are skipped)
pytest -m gpu -v -s         # the tests that need an NVIDIA GPU: run on Colab or the GPU machine
```

The normal test run fakes a second device so every GPU code path still gets exercised, and it never touches your real `~/.cache`. The GPU tests check the things only a real GPU can prove: CPU and GPU give matching results for every op at three sizes, the out-of-memory fallback works for real, the GPU load reading works, calibration works, and a long task switches to Hybrid within two chunks when the GPU is under stress.

---

## Reproducing my results

The research questions, targets and plan are in [DESIGN.md](DESIGN.md), and [REQUIREMENTS.md](REQUIREMENTS.md) tracks every requirement and where it stands. All the numbers I report come from **one fixed machine with an NVIDIA GPU**. Colab is only for development. Before you start, make sure the machine is idle and plugged in, and close any other GPU jobs.

**Step 1 — check the GPU** (on Colab or the fixed machine): run `pytest -m gpu -v -s`. All 44 GPU tests should pass.

**Step 2 — collect the data** (on the fixed machine):

| What | Command | Roughly how long | Where it goes |
|------|---------|------------------|---------------|
| Benchmark dataset (RQ1, crossover graphs) | `python bench/collect.py` | about an hour | `bench/data/<host>-<time>.csv` and `.json` |
| Experiments (RQ2, Hybrid gain) | `python bench/experiments.py all` | 20–40 minutes | `bench/results/<host>-<time>.csv` and `.json` |
| Battery saver (only on a laptop with an NVIDIA GPU) | `python bench/experiments.py rq2 --conditions battery` | about the same as RQ2 | it'll ask you to turn battery saver on |
| Dispatch overhead (RQ3) | `python eval/overhead.py` | under a minute | `eval/out/overhead.csv` |

Everything saves as it goes, so if a run dies you keep whatever it measured. CPU stress uses `stress-ng` if you have it (`apt install stress-ng`); if not, it falls back to busy Python processes and tells you which. If you want to put load on the machine by hand (handy during the demo), run `python bench/stress.py gpu` or `python bench/stress.py cpu`.

**Step 3 — make the figures and tables.** Each script reads the newest data file unless you give it a path.

| Script | What it answers | What it writes to `eval/out/` |
|--------|-----------------|-------------------------------|
| `python eval/crossover.py` | where the CPU beats the GPU, for each op | `crossover.png`, `crossover.csv` |
| `python eval/rq1.py --calibration ~/.cache/autohetero/<key>/timings.csv` | RQ1: how often the model picks the faster device (80/20 split, 5-fold, leave-one-op-out, and the model as I ship it), forest vs. tree | `rq1.csv`, `rq1_by_op.csv` |
| `python eval/rq2.py` | RQ2: time under each condition compared with always-CPU, always-GPU and the best possible choice; switches; how fast slowdowns were caught; variation between runs | `rq2.png`, `rq2.csv`, `rq2_summary.csv` |
| `python eval/hybrid_gain.py` | how much Hybrid beats GPU-only by, at each batch size (only counting runs that really used both devices) | `hybrid_gain.png`, `hybrid_gain.csv`, `hybrid_gain_summary.csv` |
| `python eval/rq3.py` | RQ3: overhead per call, calibration time, lines added to the demo | `rq3.csv` |

The targets in DESIGN.md (90% accuracy or better, under 1 ms overhead, 10–30% Hybrid gain) are things I'm testing, not promises. Report what you actually measure, including any targets you miss.

---

## The demo

```bash
pip install -e ".[demo]"
streamlit run demo/app.py
```

`demo/pipeline.py` is a small image app (blur → edges → resize → classify). Its only AutoHetero line is `autohetero.enable()`. The demo runs it three ways side by side: as written, under `force("cpu")`, and under `force("gpu")`. You get per-step times and devices, the log, and `autohetero.report()`. Flip the **GPU stress** switch and it starts a separate GPU job, so you can watch AutoHetero shift work around.

---

## What's where

```
autohetero/            the library
  __init__.py          enable, auto, force, report, retrain, ops
  dispatch.py          the @auto wrapper: features -> decide -> run -> return; force(); fallbacks
  executor.py          running on a device, chunks, the slowdown check, Hybrid
  predictor.py         the forests, per-function fits, trials, the decision cache, retrain()
  features.py          what the predictor knows about a call
  calibrate.py         first-run machine benchmark, cache, timing helpers
  monitor.py           background CPU / GPU load sampler
  history.py           per-call log, per-function totals and fits
  report.py            autohetero.report()
  devices.py           moving inputs and outputs, numpy <-> torch, model copies
  ops/                 built-in ops (image, models, analytics, science) and their benchmark inputs
bench/                 collect.py (dataset), experiments.py (RQ2, Hybrid gain), stress.py (load conditions)
eval/                  one script per figure or table; common.py holds the colours and helpers
demo/                  Streamlit demo (app.py) and the one-line app it runs (pipeline.py)
tests/                 pytest; tests marked gpu need an NVIDIA GPU
DESIGN.md              design and research plan
REQUIREMENTS.md        every requirement, its status, open questions and findings
```

---

## Contributing

Start with [DESIGN.md](DESIGN.md) — it explains the design, the scope and how I work. The short version:

- Keep the public API to `enable`, `auto`, `force`, `report` and `retrain` (plus `ops`).
- `pytest` has to pass on a machine without a GPU. Put real-GPU tests behind `@pytest.mark.gpu`.
- Fix your random seeds, and never hard-code or assert the research targets.
- If you take a shortcut, leave a comment saying so: `# shortcut: <the limit>, <when to upgrade>`.
- Every figure and table in the report comes from a script in `eval/` reading collected data. No hand-edited numbers.

---

## References

1. C.-K. Luk, S. Hong, H. Kim, "Qilin: Exploiting Parallelism on Heterogeneous Multiprocessors with Adaptive Mapping," *MICRO-42*, 2009, pp. 45–55. doi:10.1145/1669112.1669121
2. D. Grewe, Z. Wang, M. O'Boyle, "Portable Mapping of Data Parallel Programs to OpenCL for Heterogeneous Systems," *CGO*, 2013, pp. 1–10.
3. C. Augonnet, S. Thibault, R. Namyst, P.-A. Wacrenier, "StarPU: A Unified Platform for Task Scheduling on Heterogeneous Multicore Architectures," *Concurrency and Computation: Practice and Experience*, 2011.
4. ONNX Runtime documentation, "Execution Providers," https://onnxruntime.ai/docs/execution-providers/
