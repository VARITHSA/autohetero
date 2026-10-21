import csv
import math

from autohetero.ops import inputs
from eval import overhead, rq1


def synthetic_dataset(path):
    """matmul and sort where the CPU wins small sizes and the GPU wins large ones, 5 runs per size."""
    with open(path, "w", newline="") as f:
        rows = csv.writer(f)
        rows.writerow(["op", "size", "device", "elements", "cpu_load", "gpu_load", "rep", "seconds"])
        for op, sizes in (("matmul", [round(64 * 2 ** (i / 2)) for i in range(12)]), ("sort", [round(10 ** (3 + i / 3)) for i in range(12)])):
            for size in sizes:
                e = sum(math.prod(a.shape) for a in inputs.make(op, size, device="meta"))
                for device, seconds in (("cpu", 2e-9 * e), ("gpu", 2e-4 + 1e-10 * e)):
                    rows.writerows([op, size, device, e, 5.0, "", rep, seconds * (1 + rep / 100)] for rep in range(5))


def test_rq1_scores_every_evaluation(tmp_path):
    data = tmp_path / "bench.csv"
    synthetic_dataset(data)
    rq1.main(data, tmp_path / "out", calibration=data)
    with open(tmp_path / "out" / "rq1.csv") as f:
        summary = {(r["evaluation"], r["model"]): r for r in csv.DictReader(f)}
    evaluations = {"holdout 80/20", "5-fold", "leave-one-op-out", "calibration -> benchmark"}
    assert {e for e, _ in summary} == evaluations and {m for _, m in summary} == {"forest", "tree"}
    assert float(summary["5-fold", "forest"]["accuracy"]) >= 0.8  # a clean crossover is learnable
    assert float(summary["holdout 80/20", "forest"]["regret"]) >= 0
    with open(tmp_path / "out" / "rq1_by_op.csv") as f:
        assert {r["op"] for r in csv.DictReader(f)} == {"matmul", "sort"}


def test_overhead_cases_are_measured(monkeypatch):
    from autohetero import dispatch

    monkeypatch.setattr(dispatch, "_enabled", dispatch._enabled)  # measure() calls enable(): undo it afterwards
    cases, modes = overhead.measure(number=200, repeats=3)
    assert len(cases) == 6 and all(math.isfinite(s) for s in cases.values())
    assert set(modes) == {"@auto function", "built-in op"} and set(modes.values()) <= {"cpu", "gpu", "hybrid"}
