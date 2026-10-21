"""Data-analytics workloads on a table held as a float tensor (rows, columns).

filter splits along rows (results concatenate in order). sort and groupby do not split: each chunk would be
sorted or grouped on its own, which gives a wrong answer (DESIGN.md §2).
"""
import torch


def filter(table, threshold=0.5):
    """Rows whose first column is above threshold, in their original order."""
    return table[table[:, 0] > threshold]


def sort(table):
    """Rows ordered by their first column. Stable, so tied rows keep their order on every device."""
    return table[torch.argsort(table[:, 0], stable=True)]


def groupby(keys, values):
    """Sum of values per distinct key: (sorted distinct keys, sums). keys: int (N,), values: (N, ...)."""
    distinct, group = torch.unique(keys, return_inverse=True)
    sums = torch.zeros(len(distinct), *values.shape[1:], dtype=values.dtype, device=values.device)
    return distinct, sums.index_add_(0, group, values)
