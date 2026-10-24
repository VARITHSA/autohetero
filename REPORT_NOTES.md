# Report notes: interconnection-network framing

Ideas for framing AutoHetero's results with the course material on interconnection networks (Feng's design decisions, static/dynamic networks; Hwang & Briggs). Every point is a **hypothesis** until `eval/` produces the numbers behind it (DESIGN §1). Nothing here has been measured yet.

AutoHetero is the smallest case of the deck's setting: two nodes (CPU, GPU), each owning its memory, joined by one link (PCIe).

## Points that carry over

1. **The problem statement.** A CPU and GPU with separate memories are a two-node distributed-memory system; every `.to("cuda")` is a remote access over the link. Small inputs are the case where the processors spend more time waiting on the link than computing. The crossover graphs (`eval/crossover.py`) are a measurement of link cost vs. compute speed.

2. **The link is measured but unused.** By the static-network metrics, the system has bisection width 1, so the link's bandwidth is the limit. `calibrate.py` measures `h2d_bytes_per_s`, `d2h_bytes_per_s` and `launch_seconds`, but the predictor never reads them; the forest learns transfer cost only implicitly, because the timings include copies. Transfer time (bytes ÷ link speed) is op-independent, so as an explicit feature it might improve leave-one-op-out accuracy (RQ1). Untested; worth a line under RQ1 or future work.

3. **Chunking is packet switching without pipelining.** Direct execution reserves the whole path (circuit switching); chunked execution is packet switching. But `executor.py` runs each chunk as copy → compute → sync in sequence — no CUDA streams, pinned memory or `non_blocking` — so copies never overlap compute. I pay the per-packet overhead (hence `MIN_CHUNK_SECONDS`, the analogue of a minimum packet size) without the overlap. If the Hybrid gain comes out small, this is a candidate explanation.

4. **Hybrid is synchronous with central control.** Each Hybrid round ends with a barrier on both devices before the next starts; one dispatcher decides for both. The split `s_gpu / (s_gpu + s_cpu)` exists to keep the barrier wait short.

5. **"Splittable" means no data routing is needed.** The SIMD prefix-sum example works only because the network can shift data between PEs. `sort` and `groupby` aren't splittable for the same reason: their outputs depend on data on the other side. Making them splittable would need a merge/combine step across the link. This justifies the `splittable=False` default.

6. **GPU stress is a blocking problem.** A second job on one GPU and one link behaves like a shared bus: one transfer or kernel stream at a time. The RQ2 GPU-stress results measure that blocking, which also motivates the M.Tech GPU-concurrency (MPS) decision (DESIGN §13).

## What doesn't carry over

The topology material (mesh, barrel shifter, hypercube, shuffle-exchange, Omega, baseline) needs many nodes. With one GPU and multi-GPU out of scope (DESIGN §3), there is no topology to choose. It would matter only for a multi-GPU / NVLink extension.

## Before using in the report

- Back each point with a figure from `eval/out/`, or drop it.
- Cite the course slides and Hwang & Briggs (1984) for the framing.
