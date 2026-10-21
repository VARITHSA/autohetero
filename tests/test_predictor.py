import time

import pytest

from autohetero import features, predictor
from autohetero.ops import inputs

SCALARS = {"thread_seconds": 1e-5, "concat_bytes_per_s": 1e11}


def matmul(n, load=(0.0, None)):
    return features.of("matmul", inputs.make("matmul", n, device="meta"), load)


def synthetic(cpu, gpu, cpu_sizes=(64, 128, 256, 512), gpu_sizes=(64, 128, 256, 512, 1024, 2048, 4096)):
    """Training rows where seconds = cpu(elements) on the CPU and gpu(elements) on the GPU."""
    return [(matmul(n), "cpu", cpu(matmul(n)["elements"])) for n in cpu_sizes] + [
        (matmul(n), "gpu", gpu(matmul(n)["elements"])) for n in gpu_sizes]


@pytest.fixture
def fresh(monkeypatch, fake_gpu):
    monkeypatch.setattr(predictor, "_model", None)
    monkeypatch.setattr(predictor, "_cache", {})


def test_crossover_and_extrapolation_beyond_the_largest_cpu_size(fresh):
    # CPU 2 ns/element; GPU 0.2 ms launch + 0.1 ns/element: the GPU wins above ~1e5 elements (n ~ 230)
    predictor.train(synthetic(lambda e: 2e-9 * e, lambda e: 2e-4 + 1e-10 * e), SCALARS)
    assert predictor.decide("matmul", matmul(64))[0] == "cpu"
    # CPU was only measured up to n=512. Flat extrapolation would predict ~1 ms there and wrongly pick the CPU;
    # per-element extrapolation predicts ~67 ms, close to the true 2e-9 * 33.5M elements.
    mode, times, _ = predictor.decide("matmul", matmul(4096))
    assert mode == "gpu" and times["cpu"] == pytest.approx(2e-9 * matmul(4096)["elements"], rel=0.2)


def test_hybrid_only_for_splittable_calls_that_beat_the_margin(fresh):
    predictor.train(synthetic(lambda e: 1e-9 * e, lambda e: 1e-9 * e, cpu_sizes=(512, 1024, 2048)), SCALARS)
    assert predictor.decide("matmul", matmul(2048), splittable=True)[0] == "hybrid"  # equal speeds: ~2x faster
    assert predictor.decide("matmul", matmul(2048), splittable=False)[0] in ("cpu", "gpu")
    assert predictor.decide("one_row", {**matmul(2048), "batch": 1}, splittable=True)[0] in ("cpu", "gpu")
    predictor.train(synthetic(lambda e: 1e-8 * e, lambda e: 1e-9 * e, cpu_sizes=(512, 1024, 2048)), SCALARS)
    # GPU 10x faster: Hybrid is only ~9% faster than the GPU alone, under the 10% margin
    mode, times, _ = predictor.decide("matmul", matmul(2048), splittable=True)
    assert mode == "gpu" and times["hybrid"] < times["gpu"]


def test_forest_runs_only_on_a_cache_miss_and_hits_are_cheap(fresh, monkeypatch):
    predictor.train(synthetic(lambda e: 2e-9 * e, lambda e: 2e-4 + 1e-10 * e), SCALARS)
    calls = []
    predict = predictor.Model.predict
    monkeypatch.setattr(predictor.Model, "predict", lambda self, f, kind="forest": calls.append(1) or predict(self, f, kind))
    assert predictor.decide("matmul", matmul(1024))[2] is True  # new decision
    assert predictor.decide("matmul", matmul(1100))[2] is False  # same log2 size bucket
    assert predictor.decide("matmul", matmul(1024, load=(90.0, None)))[2] is True  # CPU busy: another bucket
    assert len(calls) == 2
    f = matmul(1024)
    start = time.perf_counter()
    for _ in range(10_000):
        predictor.decide("matmul", f)
    assert time.perf_counter() - start < 0.05  # under 5 us per cached decision


def test_no_gpu_always_cpu(monkeypatch):
    monkeypatch.setattr(predictor.devices, "GPU", None)
    assert predictor.decide("matmul", matmul(4096))[0] == "cpu"


def test_rows_from_a_dataset_csv(tmp_path):
    data = tmp_path / "timings.csv"
    data.write_text(
        "op,size,device,elements,cpu_load,gpu_load,rep,seconds\n"
        "sort,1000,cpu,4000,3.5,,0,0.002\nsort,1000,cpu,4000,3.5,,1,0.001\nsort,1000,cpu,4000,3.5,,2,0.009\n"
        "blur,\"(1, 3, 64, 64)\",gpu,12288,,40.0,0,0.5\n"
    )
    (f1, d1, s1), (f2, d2, s2) = predictor.rows(data)
    assert (f1["op"], f1["elements"], f1["cpu_load"], f1["gpu_load"], d1, s1) == ("sort", 4000, 3.5, None, "cpu", 0.002)
    assert (f2["op"], f2["batch"], f2["cpu_load"], f2["gpu_load"], d2, s2) == ("blur", 1, None, 40.0, "gpu", 0.5)


def test_forest_trees_stay_reachable():
    model = predictor.Model(synthetic(lambda e: 2e-9 * e, lambda e: 1e-9 * e))
    assert len(model.forests["gpu"].estimators_) == 100 and set(model.trees) == {"cpu", "gpu"}
