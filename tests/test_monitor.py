import time

import pytest
import torch

from autohetero import monitor


def test_sampler_runs_in_the_background_and_reads_are_cheap():
    monitor.start()
    monitor.start()  # idempotent: still one sampler thread
    cpu, gpu = monitor.utilisation()
    assert 0 <= cpu <= 100
    assert gpu is None if not torch.cuda.is_available() else 0 <= gpu <= 100
    time.sleep(2 * monitor.PERIOD)
    assert monitor._thread.is_alive()
    start = time.perf_counter()
    for _ in range(10_000):
        monitor.utilisation()
    assert time.perf_counter() - start < 0.1  # hot path: no syscalls, well under 10 us per read


@pytest.mark.gpu
def test_gpu_load_is_known_on_a_real_gpu():
    monitor.start()
    assert monitor.utilisation()[1] is not None
