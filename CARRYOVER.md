# AutoHetero → M.Tech carry-over

The inferences DESIGN §13 asks for, mapped to the M.Tech project review deck, *Intelligent Execution Strategy Selection* (22 slides, September 2026).

**Status: draft, 2026-10-25. No numbers yet.** Every "Measured" cell below waits for the runs on the fixed NVIDIA machine (`bench/` then `eval/`, see README). Until then, each row says what the result *can* show and which way the inference goes for each outcome. No target, mock-up or Apple-GPU rehearsal number goes in a cell.

**Rule (§13).** AutoHetero's results support only what AutoHetero measured. They can motivate an M.Tech claim, but never stand in for an M.Tech measurement. The deck's illustrative figures (slide 14's concurrency curve, slide 15's four cases) stay illustrative until the M.Tech work measures them; nothing from AutoHetero fills them.

---

## 1. Inferences, by M.Tech slide

| # | M.Tech claim (slide) | AutoHetero evidence | Inference | Measured |
|---|---|---|---|---|
| 1 | The right processor depends on the workload; small workloads and data movement make the GPU lose (3, 4). "Clear wins on small inputs" (18). | Crossover sizes per op: `eval/out/crossover.csv` | If the CPU wins below a crossover size for most ops, the same kind of work flips device with input size: direct motivation for slides 3–4 and for an input-size feature. If the GPU wins at every size tested, the problem slide must lean on concurrency (rows 2, 8) instead of small inputs. | pending |
| 2 | Jobs on one GPU compete; past a point concurrency hurts; MPS is required (4, 14). | GPU-only time under GPU stress vs idle: `eval/out/rq2_summary.csv`, condition `gpu` | AutoHetero's stress job is a second PyTorch process **without MPS**, so this measures time-sliced sharing. A large slowdown is evidence that sharing a GPU blindly costs every job, which motivates choosing the concurrency level (and using MPS) rather than maximising it. It says nothing about where the MPS peak sits. | pending |
| 3 | Scope assumes one GPU and no competing users (9), yet shared lab servers are the primary application (19). | Same `gpu` row, plus AutoHetero's speedup vs GPU-only under stress | AutoHetero's built-in decisions are trained on an idle machine and ignore load (REQUIREMENTS finding 3). If AutoHetero loses to the oracle mostly under stress, an idle-trained model is the weak point on a shared server: the "no competing users" assumption is the M.Tech's main threat to validity for slide 19 and should be stated next to it. | pending |
| 4 | Confidence as an output; a model trusted blindly fails on unfamiliar jobs (7, 13). | Leave-one-op-out vs 80/20 accuracy and regret, forest vs tree: `eval/out/rq1.csv` | If held-out-op accuracy falls well below random-split accuracy, the model is unreliable on workloads it hasn't seen, which is exactly the case for a confidence score (and for the M.Tech's held-out K-Means / K-NN evaluation). If the two are close, device choice alone rarely needs confidence, and the M.Tech case rests on near-ties between strategies (slide 13's second trigger). | pending |
| 5 | Disagreement across trees gives the confidence score (12). | The forests keep `estimators_` (PRD-3); the benchmark dataset has measured times for every size | Not computed by any `eval/` script yet. On AutoHetero's data I can check whether per-tree spread on held-out sizes rises with prediction error. If it does, that is early support for slide 12's mechanism; if it doesn't, the M.Tech needs another confidence signal. Motivation only: the M.Tech must re-test on its own workloads. | not computed |
| 6 | Measure only when unsure; calibration on a minority of jobs (7, 18). | Switch rate, time to detect, first-call cost: `eval/out/rq2.csv` (`switch_rate`, `detect_ms`, `first_call_ms`) | AutoHetero always measures a user function on its first call (the trial) and watches every chunk. If switches are rare and first calls cost a lot, most of that measuring was unnecessary: the case for measuring *only* when confidence is low. If switches fire often, measurement was needed often, and the M.Tech threshold will calibrate many jobs. | pending |
| 7 | Hybrid CPU + GPU uses a simple ratio policy (limitation, 20). | Hybrid gain vs batch size: `eval/out/hybrid_gain_summary.csv` (`gain_from_batch`) | §13 planned "drop Hybrid if the gain is small"; the deck keeps CPU + GPU but limits it to a ratio policy. If the gain is small or starts only at large batches, the ratio policy is enough and effort belongs in concurrency. If the gain is large and wide, Hybrid deserves more than a ratio policy (future work). | pending |
| 8 | "Decision takes milliseconds, the job takes seconds" (16). | Dispatch overhead per call: `eval/out/overhead.csv` (cached decision, user-function fit, forest decision miss) | AutoHetero needed a decision cache because its calls last milliseconds. The uncached forest decision is the M.Tech decision path's cost before confidence and calibration. If it is milliseconds, the M.Tech can drop caching entirely: a per-job decision is far below a job's run time. | pending |
| 9 | Predicting log time keeps large jobs from dominating; small jobs are where the CPU wins (12). | Design lesson, not a measurement: AutoHetero learns log(seconds per element) | Learning time per element lets predictions grow beyond the largest size measured instead of flattening, which matters because calibration measures few large sizes. Carry the same target over; it costs nothing. | n/a |
| 10 | Analytical baseline: startup + bytes ÷ bandwidth + work ÷ throughput (12). | Calibration measures `h2d_bytes_per_s`, `d2h_bytes_per_s`, `launch_seconds`; the forest never uses them | These are the baseline's first two terms, already measured per machine. AutoHetero gives no evidence on the baseline's accuracy; the code that measures them carries over as is. | n/a |
| 11 | Never slower than always-GPU; compare against fixed strategies and an oracle, overhead charged (17, 18). | Speedup vs GPU-only and gap to oracle (within 5%, DESIGN §1): `eval/out/rq2_summary.csv` | The method is the same at per-call scale (EVL-6: overhead inside every number). Every workload where AutoHetero is slower than always-GPU is a failure mode the M.Tech must design against; name each one. | pending |

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
| `bench/` | Stress harness, alternating strategies, recorded warm-up rep | Strategies become CPU / GPU / CPU + GPU / GPU × N via MPS |
| `dispatch.py`, `executor.py` | Not reused | Per-call chunking and slowdown switching don't apply to whole jobs |

## 4. Before this note is final

- [ ] Fill every "pending" cell from `eval/out/` on the fixed machine. Drop any inference the measured result doesn't support, and say so.
- [ ] Row 5: decide whether to add the tree-spread analysis to `eval/rq1.py`.
- [ ] Tag `autohetero-v1` on the code the M.Tech work starts from (CO-2).
- [ ] Export the benchmark dataset in the M.Tech run-log format, marked as AutoHetero data (CO-3, needs Q10).
- [ ] Cite AutoHetero as prior work per CO-4.
