import csv
import json

from bench import collect


def test_collect_writes_five_runs_per_device_and_stops_slow_series(tmp_path, fake_gpu):
    path = collect.collect(["sort"], tmp_path, max_seconds=0.05)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    first = [r for r in rows if r["size"] == "1000"]
    assert sorted(r["device"] for r in first) == ["cpu"] * 5 + ["gpu"] * 5
    assert all(0 <= float(r["cpu_load"]) <= 100 for r in rows)
    assert "10000000" not in {r["size"] for r in rows}  # sorting 10^7 rows takes far longer than 50 ms, even on a fast CPU
    assert json.loads(path.with_suffix(".json").read_text())["torch"]
