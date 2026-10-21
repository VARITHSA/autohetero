import csv
import json

import pytest

from bench import experiments
from eval import hybrid_gain, rq2, rq3


def results(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(experiments.COLUMNS)
        w.writerows(rows)


def run(condition, workload, size, strategy, rep, seconds, mode=None, slowdown="", chunks="", detect=""):
    return ["rq2", condition, workload, size, strategy, rep, seconds, mode or strategy, slowdown, chunks, detect, False]


def test_rq2_speedups_oracle_switches_and_summary(tmp_path):
    rows = []
    for rep in range(3):
        # idle matmul: GPU 1 s, Hybrid 0.8 s (the oracle), CPU 4 s; AutoHetero 1.0 s on GPU (first call 2 s)
        rows += [run("idle", "matmul", 2048, "cpu", rep, 4.0), run("idle", "matmul", 2048, "gpu", rep, 1.0),
                 run("idle", "matmul", 2048, "hybrid", rep, 0.8), run("idle", "matmul", 2048, "autohetero", rep, 2.0 if rep == 0 else 1.0, "gpu")]
        # gpu-stressed sort: CPU 1 s (oracle), GPU 3 s; AutoHetero switches after 2 chunks, 1.5 s
        rows += [run("gpu", "sort", 10**6, "cpu", rep, 1.0), run("gpu", "sort", 10**6, "gpu", rep, 3.0),
                 run("gpu", "sort", 10**6, "autohetero", rep, 1.5, "gpu", slowdown=2.4, chunks=2, detect=0.2)]
    results(tmp_path / "r.csv", rows)
    rq2.main(tmp_path / "r.csv", tmp_path / "out")
    with open(tmp_path / "out" / "rq2.csv") as f:
        table = {r["workload"]: r for r in csv.DictReader(f)}
    m, s = table["matmul"], table["sort"]
    assert float(m["oracle_ms"]) == 800 and float(m["speedup_vs_gpu"]) == 1 and float(m["gap_to_oracle"]) == 0.25
    assert float(m["first_call_ms"]) == 2000 and float(m["switch_rate"]) == 0
    assert float(s["speedup_vs_gpu"]) == 2 and float(s["switch_rate"]) == 1 and float(s["detect_ms"]) == 200
    with open(tmp_path / "out" / "rq2_summary.csv") as f:
        summary = {r["condition"]: r for r in csv.DictReader(f)}
    assert float(summary["gpu"]["within_5pct_of_oracle"]) == 0 and float(summary["idle"]["speedup_vs_cpu_geomean"]) == 4
    assert (tmp_path / "out" / "rq2.png").stat().st_size > 0


def test_hybrid_gain_finds_where_the_gain_starts(tmp_path):
    rows = []
    for n, hybrid in ((64, 1.1), (128, 0.95), (256, 0.8)):  # GPU-only takes 1 s at every size
        for rep in range(3):
            rows += [["hybrid_gain", "idle", "matmul", n, s, rep, t, s, "", "", "", False] for s, t in (("gpu", 1.0), ("hybrid", hybrid))]
    for rep in range(3):  # runs that didn't use both devices aren't Hybrid results
        rows += [["hybrid_gain", "idle", "matmul", 1, "gpu", rep, 1.0, "gpu", "", "", "", False, 1.0],
                 ["hybrid_gain", "idle", "matmul", 1, "hybrid", rep, 0.1, "cpu", "", "", "", False, 0.0],  # can't split
                 ["hybrid_gain", "idle", "matmul", 2, "gpu", rep, 1.0, "gpu", "", "", "", False, 1.0],
                 ["hybrid_gain", "idle", "matmul", 2, "hybrid", rep, 0.1, "hybrid", "", "", "", False, 0.0]]  # all on CPU
    results(tmp_path / "r.csv", rows)
    hybrid_gain.main(tmp_path / "r.csv", tmp_path / "out")
    with open(tmp_path / "out" / "hybrid_gain_summary.csv") as f:
        (s,) = csv.DictReader(f)
    assert s["gain_from_batch"] == "128" and float(s["max_gain"]) == pytest.approx(0.2) and s["max_gain_batch"] == "256"
    assert (tmp_path / "out" / "hybrid_gain.png").stat().st_size > 0


def test_rq3_collects_overhead_calibration_and_lines(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "overhead.csv").write_text("case,microseconds_per_call\n\"@auto, cached decision\",8.67\n")
    cal = tmp_path / "calibration.json"
    cal.write_text(json.dumps({"seconds": 27.2}))
    rq3.main(out, cal)
    with open(out / "rq3.csv") as f:
        values = {r["metric"]: r["value"] for r in csv.DictReader(f)}
    assert values == {"dispatch overhead: @auto, cached decision": "8.67", "calibration, once per machine": "27.2",
                      "lines added to the demo app": "1"}


def test_eval_scripts_skip_what_a_dead_run_left_incomplete(tmp_path):
    rows = [run("idle", "sort", 1000, s, rep, 1.0) for rep in range(2) for s in ("cpu", "gpu", "autohetero")]
    rows += [run("idle", "matmul", 64, "cpu", 0, 1.0), run("idle", "matmul", 64, "gpu", 0, 1.0)]  # died here
    rows += [["hybrid_gain", "idle", "matmul", 64, "gpu", 1, 1.0, "gpu", "", "", "", False]]  # and here
    results(tmp_path / "r.csv", rows)
    rq2.main(tmp_path / "r.csv", tmp_path / "out")
    hybrid_gain.table(tmp_path / "r.csv")
    with open(tmp_path / "out" / "rq2.csv") as f:
        assert [r["workload"] for r in csv.DictReader(f)] == ["sort"]
