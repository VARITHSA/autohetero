import pytest

from autohetero import dispatch, predictor
from bench import experiments


@pytest.fixture
def runs(monkeypatch, fake_gpu):
    """Collect the rows the experiments write; the predictor always says GPU."""
    monkeypatch.setattr(dispatch, "_unhealthy", set())
    monkeypatch.setattr(dispatch, "_enabled", True)
    times = lambda name, f, splittable=False, quiet=False: ("gpu", {"cpu": 1e-6 * f["batch"], "gpu": 1e-6 * f["batch"]}, False)
    monkeypatch.setattr(predictor, "decide", times)
    rows = []
    return rows


def test_rq2_runs_every_strategy_alternately_with_a_warm_up(runs):
    experiments.rq2(runs.append, ["idle"], reps=2, workloads=[("sort", 1000), ("matmul", 64), ("enhance", (4, 3, 32, 32))])
    by = lambda workload: [(r[4], r[5]) for r in runs if r[2] == workload]
    assert by("sort") == [(s, rep) for rep in range(3) for s in ("cpu", "gpu", "autohetero")]  # sort can't split
    assert [s for s, rep in by("matmul") if rep == 0] == ["cpu", "gpu", "hybrid", "autohetero"]
    assert all(len(r) == len(experiments.COLUMNS) and r[6] > 0 for r in runs)
    auto = [r for r in runs if r[4] == "autohetero"]
    assert {r[7] for r in auto} == {"gpu"} and {r[7] for r in runs if r[4] == "cpu"} == {"cpu"}
    assert {r[12] for r in runs if r[4] == "cpu"} == {0.0} and {r[12] for r in runs if r[4] == "gpu"} == {1.0}


def test_hybrid_gain_compares_gpu_and_hybrid_until_runs_get_slow(runs):
    experiments.hybrid_gain(runs.append, reps=1, max_seconds=60, sizes={"matmul": [64, 128]})
    matmul = [(r[3], r[4]) for r in runs if r[2] == "matmul"]
    assert matmul == [(n, s) for n in (64, 128) for _ in range(2) for s in ("gpu", "hybrid")]
