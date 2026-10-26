# AutoHetero → M.Tech carry-over

The inferences DESIGN §13 asks for, mapped to the M.Tech project review deck, *Intelligent Execution Strategy Selection* (22 slides, September 2026).

**Status: measured, 2026-10-26.** Every number below comes from the fixed machine: an RTX 3050 Laptop GPU (4 GB, 60 W) and a Ryzen 7 5800H (8 cores, 16 threads), 16 GB RAM, Windows, PyTorch 2.11 + CUDA 12.8. The sources are `eval/out/` and `eval/results.ipynb`. RQ2 figures are geometric means over workloads with sort 10M left out, because its always-GPU baseline is inflated by run order (row 12).

**Rule (§13).** AutoHetero's results support only what AutoHetero measured. They can motivate an M.Tech claim, but never stand in for an M.Tech measurement. The deck's illustrative figures (slide 14's concurrency curve, slide 15's four cases) stay illustrative until the M.Tech work measures them; nothing from AutoHetero fills them.

---

## 1. Inferences, by M.Tech slide

| # | M.Tech claim (slide) | Measured on AutoHetero | Inference | Verdict |
|---|---|---|---|---|
| 1 | The right processor depends on the workload; small workloads and data movement make the GPU lose (3, 4). "Clear wins on small inputs" (18). | The CPU is faster below a crossover size for 7 of 10 ops (blur, resize, sobel, filter, sort, groupby, matmul). The GPU wins at every size for resnet18 and mobilenetv2 (up to 19.5× and 13.2×). The CPU wins at every size for fft (GPU only 0.44–0.89× as fast). `crossover.csv` | The faster device flips with input size for most ops, and for one op "use the GPU" is wrong at every size. Direct motivation for slides 3–4 and for input size as a feature. | Supported |
| 2 | Jobs on one GPU compete; past a point concurrency hurts; MPS is required (4, 14). | With a second process on the GPU (time-sliced, no MPS), GPU-only runs took **1.68×** as long (16 workloads, 1.2–2.3×). `rq2.csv`, `gpu` vs `idle` | One time-sliced neighbour cost a job about 40% of its speed. This is the victim job's time only, not total throughput, and says nothing about MPS: slide 14's curve stays illustrative. | Supported (motivation) |
| 3 | Scope assumes one GPU and no competing users (9), yet shared lab servers are the primary application (19). | AutoHetero vs always-GPU: **0.82×** under GPU stress, **0.21×** under CPU stress. Under CPU stress the GPU chunks looked slow, so the switch moved the rest of the call to Hybrid, onto the loaded CPU: resnet18 at batch 128 took 8.97 s against 0.37 s GPU-only. | A decision trained on an idle machine fails badly once other users load the CPU. On the shared servers of slide 19, "no competing users" is the main threat to validity: state it next to slide 19, or make the decision load-aware. | Supported, strongly |
| 4 | Confidence as an output; a model trusted blindly fails on unfamiliar jobs (7, 13). | Forest accuracy **91.1%** on 5-fold (4.3% regret), 88% on an 80/20 split, **80.7%** trained on the 33 s calibration alone, **68.8%** on leave-one-op-out (tree 74.4%). Worst unseen op: fft, 14%. `rq1.csv`, `rq1_by_op.csv` | Accuracy falls from 91% to 69% on kinds of work the model hasn't seen, and the worst miss is the op least like the others. That is the case for a confidence score and for the M.Tech's held-out K-Means / K-NN test. | Supported |
| 5 | Disagreement across trees gives the confidence score (12). | Not computed. The forests keep `estimators_`, and the dataset and calibration are in the repo, so it can be computed now. | Still open: whether per-tree spread rises with prediction error on held-out sizes. | Not computed |
| 6 | Measure only when unsure; calibration on a minority of jobs (7, 18). | AutoHetero watches every splittable call in chunks. At idle, with nothing to react to, it ran at **0.70×** always-GPU, and switches still fired in 10% of runs (fft 4096: 80%). Under GPU stress switches fired in 34% of runs and were detected in a median 74 ms, but paid off on few workloads (mobilenetv2 batch 128: 1.20×). The user-function trial was cheap: `enhance`'s first call took 287 ms against a 234 ms median. `rq2_summary.csv`, `rq2.csv` | Measuring on every call cost about 30% at idle, where it bought nothing, and paid off only for a few workloads under contention. That supports measuring only when unsure. The M.Tech probes before a job rather than chunking it, so the chunking cost itself doesn't carry over; the frequency argument does. | Supported |
| 7 | Hybrid CPU + GPU uses a simple ratio policy (limitation, 20). | Hybrid beat GPU-only only on resize (+11% at batch 256), filter (+1% at 10M rows) and mobilenetv2 at batch 2 (+19%, no larger batch). Most lose: resnet18 −64%, sobel −77%, blur −22%, matmul −17%, fft −10%. AutoHetero's Hybrid estimate also over-promised: it picked Hybrid for mobilenetv2 at batch 32, which took 165 ms against 60 ms GPU-only. `hybrid_gain_summary.csv` | On a laptop CPU next to an RTX 3050, CPU + GPU rarely helps, and the ideal-overlap estimate T_cpu·T_gpu / (T_cpu + T_gpu) is too optimistic. Likely because the CPU also feeds the GPU and both share one power budget (not isolated in this run). Keep CPU + GPU behind a measurement, or leave it out. Slide 15's case C (CPU + GPU beating GPU) is the kind of result this machine didn't show. | Supported: keep it simple or drop it |
| 8 | "Decision takes milliseconds, the job takes seconds" (16). | A cached decision costs 0.16 ms per call, one from a function's fit 0.10 ms, an uncached forest decision **26.9 ms**. Calibration: **32.6 s**, once per machine. `overhead.csv`, `rq3.csv` | A forest decision is about 27 ms on this laptop, negligible next to a job lasting seconds, so the M.Tech decision path needs no cache. The ~33 s calibration is the larger one-off cost to budget. | Supported |
| 9 | Predicting log time keeps large jobs from dominating; small jobs are where the CPU wins (12). | Design lesson, not a measurement: AutoHetero learns log(seconds per element). | Learning time per element lets predictions grow beyond the largest size measured instead of flattening. Carry the same target over. | n/a |
| 10 | Analytical baseline: startup + bytes ÷ bandwidth + work ÷ throughput (12). | Calibration measured host→GPU **6.4 GB/s**, GPU→host **3.4 GB/s**, launch **65 µs**. `bench/calibration/` | These are the baseline's first two terms, measured per machine: a 100 MB batch costs ~16 ms to send and ~30 ms to return. No evidence yet on the baseline's accuracy. | Inputs measured |
| 11 | Never slower than always-GPU; compare against fixed strategies and an oracle, overhead charged (17, 18). | AutoHetero vs always-GPU: 0.70× idle, 0.82× GPU stress, 0.21× CPU stress, 0.65× on battery. Within 5% of the oracle on 4 of 17, 6 of 16, 4 of 15 and 4 of 16 workloads. Against always-CPU: 3.3×, 2.3×, 1.9×, 3.0× faster. | AutoHetero did not meet "never slower than always-GPU", and it names the failure modes the M.Tech must design against: monitoring overhead with nothing to react to; switching onto a loaded device; mispredicting an op unlike the training set (fft); an optimistic CPU + GPU estimate. Claim A is strong; this shows how easily a dispatcher falls short of always-GPU. | Not met by AutoHetero |
| 12 | Evaluation method: fixed strategies vs the oracle (17). | Method lesson: in RQ2 the forced GPU run always came right after a long CPU run. Rechecked for sort 10M: whichever run comes right after a 3.4 s CPU sort takes ~260–270 ms; the same GPU sort alone takes ~91 ms. `bench/results/recheck/` | Run order biases the baselines. The M.Tech evaluation should rotate or shuffle the strategy order per repetition, and warm up each strategy after a long run on the other device. | Lesson for slide 17 |

**Limits of this run.** The 4 GB card skipped the 64 × 3 × 1080 × 1920 image workloads everywhere, and matmul 8192 outside idle. CPU stress used Python busy loops (no `stress-ng` on Windows). On the battery run, Windows reported Energy saver as off, although it was switched on by hand. The executor constants were untuned first guesses. The week-6 exit check passed only when the predictor picked GPU at idle (43 of 44 GPU tests passed).

## 2. How far the workloads overlap

| | AutoHetero (PyTorch, in-process, per call) | M.Tech (C / CUDA C binaries, per job) |
|---|---|---|
| Matrix multiply | `matmul` | training workload |
| Convolution | `blur`, `sobel` (conv2d), `resnet18`, `mobilenetv2` | training workload |
| Reduction | none directly (`groupby` is the nearest) | training workload |
| K-Means, K-NN | none | held-out test workloads |

Only matrix multiply and convolution overlap, and the runtimes differ (PyTorch ops timed in-process vs binaries that time their own compute region). So **trends** carry over (crossovers exist, sharing costs time, held-out ops are harder), **absolute times don't**.

## 3. Planned reuse (to confirm once the M.Tech code exists)

| AutoHetero module | Reuse in the M.Tech work | What changes |
|---|---|---|
| `calibrate.py` | Bandwidth and launch-latency measurements, machine-keyed cache | Feeds the analytical baseline (row 10) |
| `features.py` | The feature-vector idea: size, bytes, batch, dtype, load | Adds compute intensity, access pattern and occupancy from a probe run (slide 12) |
| `predictor.py` | Random forest on log time per device, `estimators_` reachable | One model per strategy including GPU × N; confidence from tree spread; a ratio model; no decision cache |
| `history.py` | Predicted logged next to measured for every run | The gap becomes the data for tuning the confidence threshold (slide 13) |
| `monitor.py` | CPU / GPU utilisation sampling | Unchanged |
| `bench/` | Stress harness, alternating strategies, recorded warm-up rep | Strategies become CPU / GPU / CPU + GPU / GPU × N via MPS; rotate their order per repetition (row 12) |
| `dispatch.py`, `executor.py` | Not reused | Per-call chunking and slowdown switching don't apply to whole jobs |

## 4. What this changes in the M.Tech deck

- Slide 9 / 19: name "no competing users" as the main threat to validity for shared servers (row 3).
- Slide 15, case C: CPU + GPU winning is illustrative and wasn't seen on this laptop (row 7); expect CPU + GPU to need a measurement before it is chosen.
- Slide 17: rotate the strategy order per repetition (row 12).
- Slide 18: "never slower than always-GPU" is the hardest expectation; AutoHetero missed it in every condition (row 11).

## 5. Before this note is final

- [x] Fill the measured cells from `eval/out/` on the fixed machine (2026-10-26).
- [ ] Row 5: decide whether to add the tree-spread analysis to `eval/rq1.py`.
- [ ] Tag `autohetero-v1` on the code the M.Tech work starts from (CO-2).
- [ ] Export the benchmark dataset in the M.Tech run-log format, marked as AutoHetero data (CO-3, needs Q10).
- [ ] Cite AutoHetero as prior work per CO-4.
