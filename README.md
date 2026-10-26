# AutoHetero

![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange) ![License](https://img.shields.io/badge/license-MIT-green)

**AutoHetero runs each PyTorch call on the CPU, the GPU, or both — whichever is fastest.**

A GPU isn't always the faster choice. For small inputs, moving data to the GPU costs more than the work itself, and a GPU that's busy with another job can end up slower than the CPU. I built AutoHetero to make that decision automatically: it benchmarks your machine once, predicts the fastest device for every call, and shifts work to the CPU when the GPU falls behind.

## Features

- **One line to enable.** No configuration, no `.to("cuda")`.
- **Tensors or NumPy.** Results come back in the same type and on the same device you passed in.
- **Hybrid mode.** Large batches can be split across the CPU and GPU at the same time.
- **Adapts while running.** If the GPU slows down mid-call, the rest of the work is rebalanced.
- **Safe fallback.** GPU errors (like running out of memory) rerun the call on the CPU instead of crashing.
- **Transparent.** A short log line for every decision, and `autohetero.report()` shows what ran where.

## Requirements

- Python 3.11+ and PyTorch 2.x
- An NVIDIA GPU with a CUDA build of PyTorch. Without one, AutoHetero still works and runs everything on the CPU.

## Installation

AutoHetero isn't on PyPI yet, so install it from GitHub.

**To use it in your own project**, install it into that project's environment:

```bash
pip install git+https://github.com/VARITHSA/autohetero.git
```

**To run the demos, tests and benchmarks**, clone the repository:

```bash
git clone https://github.com/VARITHSA/autohetero.git
cd autohetero
pip install -e .            # add ".[dev]" for tests, ".[demo]" for the demo
```

Check that PyTorch can see your GPU (it should print `True`):

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints `False`, install the CUDA build of PyTorch for your driver from [pytorch.org](https://pytorch.org/get-started/locally/).

## Try it

From the cloned repository:

1. **With and without, side by side.** This runs the same image pipeline as plain PyTorch on the CPU, by hand on the GPU, and with AutoHetero, checks that all three give the same answer, and prints their times:

   ```bash
   python demo/compare.py
   ```

2. **The demo app.** Install the demo extra, then start the app; it opens in your browser:

   ```bash
   pip install -e ".[demo]"
   ```

   ```bash
   streamlit run demo/app.py
   ```

   Pick a batch size and resolution and click **Run**. Turn on **GPU stress** to watch AutoHetero move work to the CPU. The **playground** page in the sidebar runs any built-in op at any size.

3. **The measured results.** Open `demo/showcase.ipynb` in Jupyter (`pip install jupyter` first). It reads the results from my RTX 3050 runs, so it works on a machine without a GPU.

## Using it in your code

1. Import AutoHetero and call `autohetero.enable()` once, at the start of your program.
2. Use the built-in ops (`autohetero.ops.*`), or put `@autohetero.auto` on your own functions.
3. Keep your data on the CPU, as usual. Don't call `.to("cuda")`; AutoHetero moves the data and brings the result back.
4. Call `autohetero.report()` at the end to see where each call ran.

**Built-in ops** — one line:

```python
import torch
import autohetero

autohetero.enable()                      # the one line: built-in ops now run where they're fastest

images = torch.rand(16, 3, 720, 1280)    # on the CPU, as usual
edges = autohetero.ops.sobel(autohetero.ops.blur(images))   # each op is dispatched; the result comes back on the CPU
```

**Your own functions** — one decorator:

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

Models work too. Pass them in as arguments:

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

**Manual override**, when you know better:

```python
with autohetero.force("cpu"):      # or "gpu", or "hybrid"
    labels = classify(batch, model)
```

**See what happened:**

```python
autohetero.report()
```

```
function  calls  modes                time  saved vs GPU (est.)  fallbacks  switches
--------  -----  ---------------  --------  -------------------  ---------  --------
blur          3  GPU 3              91.2 ms             12.0 ms          0         0
classify      2  HYBRID 1, GPU 1   612.0 ms             48.0 ms          0         1
```

The first call on a new machine runs a one-time calibration (about 30 seconds).

## How it works

1. **Calibrate.** On first use, AutoHetero times its built-in ops on the CPU and GPU and caches the results in `~/.cache/autohetero/`.
2. **Predict.** A small learned model (random forests) estimates how long each call will take on each device. Your own functions are predicted from their own past runs; the first call measures both devices.
3. **Run.** Long calls run in chunks. If a chunk takes more than 1.5× longer than expected, the rest of the call is split between the CPU and GPU by their measured speeds.
4. **Learn.** Every call is logged with its predicted and actual time, so predictions improve over time. Call `autohetero.retrain()` to update the model.

The full design is in [DESIGN.md](https://github.com/VARITHSA/autohetero/blob/main/DESIGN.md).

## API

| Function | What it does |
|----------|--------------|
| `autohetero.enable()` | Turns on dispatch for the built-in ops and calibrates the machine (once). |
| `@autohetero.auto(splittable=False, batch_dim=0)` | Runs every call of a function on the fastest option. Set `splittable=True` only if each output row depends only on the matching input row; that allows chunking and Hybrid mode. |
| `autohetero.force(mode)` | Context manager: run on `"cpu"`, `"gpu"` or `"hybrid"`. Applies to the current thread only. |
| `autohetero.report()` | Prints (and returns) what ran where, total time, and the estimated time saved. |
| `autohetero.retrain()` | Updates the prediction model from the calls recorded so far. |

### Tips for `@auto` functions

- Use torch operations — inputs arrive as tensors, even if you passed NumPy arrays.
- Don't modify inputs in place (`x += 1`).
- Pass models as arguments, not as globals.
- If the GPU fails, the call reruns on the CPU, so avoid side effects you can't repeat.

## Built-in ops

| Area | Ops | Can be split across devices |
|------|-----|-----------------------------|
| Images | `blur`, `resize`, `sobel` | yes |
| Models | `resnet18`, `mobilenetv2` (random weights, for benchmarking) | yes |
| Data | `filter`, `sort`, `groupby` | `filter` only |
| Science | `matmul`, `fft` | yes |

## Configuration

There's nothing to configure. Two options if you need them:

- `AUTOHETERO_QUIET=1` turns off the log lines.
- Delete `~/.cache/autohetero/` to recalibrate. (It also recalibrates on its own when the GPU, driver or PyTorch version changes.)

## Tests

```bash
pip install -e ".[dev]"
pytest                  # runs without a GPU
pytest -m gpu           # needs an NVIDIA GPU
```

## Benchmarks

The evaluation scripts reproduce the results for the project's three research questions. Run them on a machine with an NVIDIA GPU:

```bash
python bench/collect.py              # benchmark dataset
python bench/experiments.py all      # idle / GPU-stress / CPU-stress experiments
python eval/overhead.py              # dispatch overhead
python eval/crossover.py             # then the figures and tables, written to eval/out/
python eval/rq1.py
python eval/rq2.py
python eval/hybrid_gain.py
python eval/rq3.py
```

[REQUIREMENTS.md](https://github.com/VARITHSA/autohetero/blob/main/REQUIREMENTS.md) lists every requirement and its current status.

## Project structure

```
autohetero/   the library (dispatch, prediction, execution, built-in ops)
bench/        benchmark collection, experiments, load generators
eval/         scripts that produce the figures and tables
demo/         Streamlit demo
tests/        test suite
```

## Limitations

- Built for one machine with one NVIDIA GPU, for inference and data processing (not training).
- Hybrid mode only applies to splittable calls with at least 2 items.
- Very short calls aren't chunked, so they can't be rebalanced mid-call.
- Each call adds about 10 µs of overhead (a few ms the first time a new input size is seen).

## License

MIT — see [LICENSE](https://github.com/VARITHSA/autohetero/blob/main/LICENSE).
