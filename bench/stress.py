"""Load conditions for the RQ2 experiments (DESIGN.md §10), each applied by other processes.

    python bench/stress.py gpu [--seconds S]      # a parallel PyTorch job keeps the GPU busy
    python bench/stress.py cpu [--workers N]      # stress-ng --cpu N keeps the cores busy

In code:  with stress.gpu(): ...    with stress.cpu() as tool: ...
Battery saver can't be scripted: switch it on by hand, on a laptop with an NVIDIA GPU (§10).
"""
import argparse
import contextlib
import os
import shutil
import subprocess
import sys
import time

GPU_JOB = """
import torch
a = torch.rand({size}, {size}, device="{device}")
(a @ a).sum().item()
print("ready", flush=True)
while True:
    (a @ a).sum().item()  # waiting for each multiply keeps the GPU queue full but short, as a real job's would be
"""


@contextlib.contextmanager
def gpu(device="cuda", size=4096):
    """Run the block while another process keeps device busy with matrix multiplies."""
    job = subprocess.Popen([sys.executable, "-c", GPU_JOB.format(size=size, device=device)], stdout=subprocess.PIPE, text=True)
    try:
        if job.stdout.readline().strip() != "ready":  # returns once the GPU is up and busy, or the job has died
            raise RuntimeError(f"the GPU stress job failed to start (exit code {job.wait()})")
        yield
    finally:
        _stop([job])


@contextlib.contextmanager
def cpu(workers=None):
    """Run the block while other processes keep workers cores busy (all by default). Yields the tool used."""
    workers = workers or os.cpu_count()
    if shutil.which("stress-ng"):
        commands, tool = [["stress-ng", "--cpu", str(workers), "--quiet"]], "stress-ng"
    else:  # the experiment record names the tool, so a fallback can't pass for stress-ng
        commands, tool = [[sys.executable, "-c", "while True: pass"]] * workers, "python busy loops (no stress-ng)"
    jobs = [subprocess.Popen(c, stdout=subprocess.DEVNULL) for c in commands]
    try:
        time.sleep(0.5)  # let the load build up
        yield tool
    finally:
        _stop(jobs)


def _stop(jobs):
    for job in jobs:
        job.terminate()  # SIGTERM: stress-ng passes it on to its workers, SIGKILL wouldn't
    for job in jobs:
        try:
            job.wait(timeout=5)
        except subprocess.TimeoutExpired:
            job.kill()
            job.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=("gpu", "cpu"))
    parser.add_argument("--seconds", type=float, help="stop after this long (default: Ctrl-C)")
    parser.add_argument("--workers", type=int, help="cpu: busy cores (default: all)")
    parser.add_argument("--device", default="cuda", help="gpu: torch device (default: cuda)")
    a = parser.parse_args()
    load = gpu(a.device) if a.kind == "gpu" else cpu(a.workers)
    with load as tool:
        print(f"{a.kind} stress on{f' ({tool})' if tool else ''}; Ctrl-C to stop", flush=True)
        try:
            time.sleep(a.seconds if a.seconds else 1e9)
        except KeyboardInterrupt:
            pass
