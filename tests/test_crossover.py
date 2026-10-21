import csv

from eval import crossover


def test_crossover_finds_where_each_device_wins(tmp_path):
    data = tmp_path / "run.csv"
    with open(data, "w", newline="") as f:
        rows = csv.writer(f)
        rows.writerow(["op", "size", "device", "elements", "rep", "seconds"])
        # CPU wins on 10 and 100 elements, GPU from 1000 up; the CPU stopped before 10000.
        for size, cpu, gpu in [(10, 1, 5), (100, 2, 5), (1000, 30, 6), (10000, None, 8)]:
            for device, seconds in (("cpu", cpu), ("gpu", gpu)):
                if seconds is not None:
                    rows.writerows(["matmul", size, device, size, rep, seconds * (1 + rep / 100)] for rep in range(5))
    png = crossover.main(data, tmp_path / "out")
    assert png.stat().st_size > 0
    with open(tmp_path / "out" / "crossover.csv") as f:
        (summary,) = csv.DictReader(f)
    assert summary["sizes_on_both"] == "3"
    assert summary["cpu_faster_up_to_elements"] == "100" and summary["gpu_faster_from_elements"] == "1000"
    assert summary["gpu_speedup_max"] == "5"  # 30 / 6 at 1000 elements
    assert crossover.cpu_faster_spans([(1000, 30, 6), (10, 1, 5), (100, 2, 5)]) == [(10, 10 ** 1.5), (10 ** 1.5, 10 ** 2.5)]
