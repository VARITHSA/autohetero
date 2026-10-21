"""AutoHetero: per-call CPU / GPU / Hybrid dispatch for PyTorch workloads.

    import autohetero
    autohetero.enable()             # Level 1: also dispatch the built-in autohetero.ops
    @autohetero.auto                # Level 2: one function
    with autohetero.force("cpu"):   # Level 3: manual override
    autohetero.report()             # what ran where, time taken, estimated time saved
    autohetero.retrain()            # learn from this machine's measured calls
"""
from . import ops
from .dispatch import auto, enable, force
from .predictor import retrain
from .report import report

__all__ = ["auto", "enable", "force", "ops", "report", "retrain"]
