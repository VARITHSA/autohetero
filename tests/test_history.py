import json

import pytest
import torch

import autohetero
from autohetero import calibrate, devices, dispatch, executor, history, predictor


def entry(name="f", op="user", mode="gpu", seconds=1.0, predicted=None, pieces=None, fallback=False, slowdown=None, elements=100):
    return {"name": name, "op": op, "elements": elements, "bytes": 4 * elements, "batch": 10, "dtype": "float32",
            "cpu_load": 5.0, "gpu_load": None, "mode": mode, "forced": False, "predicted": predicted or {},
            "seconds": seconds, "pieces": pieces or [], "fallback": fallback, "slowdown": slowdown,
            "chunks_before_switch": None}


def log_lines():
    history.flush()
    return [json.loads(line) for line in (calibrate.folder() / "history.jsonl").read_text().splitlines()]


def test_totals_and_the_log_keep_predicted_next_to_measured():
    history.record(entry(op="sort", seconds=1.0, predicted={"cpu": 3.0, "gpu": 1.5}))  # a built-in op's prediction always counts
    history.record(entry(mode="cpu", seconds=2.0, fallback=True, slowdown=2.5))
    t = history.totals()["f"]
    assert (t["calls"], t["seconds"], t["mode_gpu"], t["mode_cpu"], t["fallbacks"], t["switches"]) == (2, 3.0, 1, 1, 1, 1)
    assert t["saved_vs_gpu_estimate"] == 0.5  # only the call with a prediction counts: 1.5 predicted - 1.0 measured
    first = log_lines()[0]
    assert first["predicted"] == {"cpu": 3.0, "gpu": 1.5} and first["seconds"] == 1.0 and "time" in first


def test_a_user_functions_first_call_is_a_guess_left_out_of_the_estimate():
    history.record(entry(seconds=1.0, predicted={"cpu": 3.0, "gpu": 9.0}, pieces=[["gpu", 100, 1.0]]))  # nothing measured yet
    history.record(entry(seconds=1.0, predicted={"cpu": 3.0, "gpu": 1.5}, pieces=[["gpu", 100, 1.0]]))  # from its own fit
    t = history.totals()["f"]
    assert (t["predicted_calls"], t["saved_vs_gpu_estimate"]) == (1, 0.5)


def test_fit_learns_a_power_law_and_follows_change():
    for e in (1e3, 1e4, 1e5, 1e6):
        history.record(entry(pieces=[["gpu", e, 1e-8 * e**1.2]]))
    assert history.predict("f", "gpu", 1e7) == pytest.approx(1e-8 * 1e7**1.2, rel=0.05)
    assert history.predict("f", "cpu", 1e7) is None  # never ran there
    for _ in range(60):  # the GPU gets 3x slower (contention): old points fade
        history.record(entry(pieces=[["gpu", 1e5, 3e-8 * 1e5**1.2]]))
    assert history.predict("f", "gpu", 1e5) == pytest.approx(3e-8 * 1e5**1.2, rel=0.1)


def test_one_size_assumes_linear_and_noise_cannot_explode_the_slope():
    history.record(entry(name="one", pieces=[["cpu", 1000, 0.01]]))
    assert history.predict("one", "cpu", 4000) == pytest.approx(0.04)
    history.record(entry(name="noisy", pieces=[["cpu", 1000, 0.001]]))
    history.record(entry(name="noisy", pieces=[["cpu", 1100, 0.01]]))  # 10x time for 10% more data
    assert history.predict("noisy", "cpu", 2000) < 0.01 * (2000 / 1100) ** history.SLOPE[1] * 1.01


def test_fits_carry_over_to_the_next_session():
    history.record(entry(pieces=[["cpu", 1000, 0.01]]))
    history.save_fits()
    history.reset()  # a new session on the same machine
    assert history.predict("f", "cpu", 4000) == pytest.approx(0.04)
    assert history.totals() == {}  # totals are per session
    (calibrate.folder() / "fits.json").write_text("{not json")
    history.reset()
    assert history.predict("f", "cpu", 4000) is None  # an unreadable file starts empty instead of crashing


@pytest.fixture
def predicts(monkeypatch, fake_gpu):
    def use(mode):
        decide = lambda name, f, splittable=False, quiet=False: (mode, {"cpu": 1e-3 * f["batch"], "gpu": 1e-3 * f["batch"]}, False)
        monkeypatch.setattr(predictor, "decide", decide)
    monkeypatch.setattr(dispatch, "_unhealthy", set())
    return use


def test_dispatched_calls_are_recorded_once_with_their_pieces(predicts):
    predicts("hybrid")
    inner = autohetero.auto(lambda x: x + 1)
    outer = autohetero.auto(splittable=True)(lambda x: inner(x) * 2)
    outer(torch.zeros(10))
    (r,) = log_lines()  # the nested call is part of the outer one, not a record of its own
    assert r["name"].endswith("<lambda>") and r["mode"] == "hybrid" and r["slowdown"] is None
    assert {p[0] for p in r["pieces"]} == {"cpu", "gpu"} and sum(p[1] for p in r["pieces"]) == pytest.approx(10)


def test_a_fallback_is_recorded_with_the_cpu_rerun_as_its_piece(predicts):
    predicts("gpu")

    @autohetero.auto
    def fragile(x):
        if executor.running_on.get() == devices.GPU:
            raise RuntimeError("CUDA error: out of memory")
        return x

    fragile(torch.zeros(4))
    (r,) = log_lines()
    assert r["fallback"] and r["mode"] == "gpu" and [p[0] for p in r["pieces"]] == ["cpu"]


def test_retrain_uses_calibration_plus_measured_builtin_calls(monkeypatch, tmp_path):
    history.record(entry(name="sort", op="sort", elements=4000, pieces=[["cpu", 4000, 0.002], ["gpu", 2000, 0.001]]))
    history.record(entry(name="mine", op="user", pieces=[["cpu", 100, 0.5]]))
    timings = tmp_path / "timings.csv"
    timings.write_text("op,size,device,elements,cpu_load,gpu_load,rep,seconds\nsort,1000,cpu,4000,,,0,0.002\n")
    monkeypatch.setattr(calibrate, "calibration", lambda: {"timings": str(timings), "thread_seconds": 0, "concat_bytes_per_s": 1})
    trained = []
    monkeypatch.setattr(predictor, "train", lambda rows, scalars: trained.extend(rows))
    autohetero.retrain()
    assert len(trained) == 3  # 1 calibration row + 2 pieces of the sort call; the user function is left to its fit
    gpu_piece = trained[-1][0]
    assert (gpu_piece["op"], gpu_piece["elements"], gpu_piece["bytes"], gpu_piece["batch"]) == ("sort", 2000, 8000, 5)


def test_the_log_rotates_instead_of_growing_forever(monkeypatch):
    monkeypatch.setattr(history, "MAX_LOG_BYTES", 2000)
    for _ in range(30):
        history.record(entry())
    history.flush()
    current, old = calibrate.folder() / "history.jsonl", calibrate.folder() / "history.old.jsonl"
    assert old.exists() and current.stat().st_size <= 2000 + 1000 and old.stat().st_size <= 2000 + 1000
    assert history.totals()["f"]["calls"] == 30  # totals are not affected
    history.record(entry(name="sort", op="sort", pieces=[["cpu", 100, 0.01]]))
    assert len(history.rows()) == 1  # retrain reads across the rotation


def test_retrain_skips_a_line_cut_short_by_a_crash():
    history.record(entry(name="sort", op="sort", pieces=[["cpu", 100, 0.01]]))
    history.flush()
    with open(calibrate.folder() / "history.jsonl", "a") as f:
        f.write('{"time": 1, "name": "sort", "op": "so')  # the process died mid-write
    assert len(history.rows()) == 1
