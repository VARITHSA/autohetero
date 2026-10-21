"""RQ1 (DESIGN.md §1, §10): how often the model picks the faster device for sizes it hasn't seen.

    python eval/rq1.py                          # newest bench/data/*.csv -> eval/out/rq1.csv, rq1_by_op.csv
    python eval/rq1.py bench/data/<file>.csv --calibration ~/.cache/autohetero/<key>/timings.csv

Scores, for the random forest and the decision tree:
  accuracy    share of held-out sizes where the predicted faster device was the measured faster one
  regret      extra time from following the predictions instead of the oracle (best measured device per size)
  time_error  median |predicted - measured| / measured, over both devices
A size's CPU and GPU runs always land in the same split. Leave-one-op-out trains without an op and tests on
it: how the model meets workloads it has never seen. --calibration trains on a 30 s calibration and tests on
the benchmark: the model AutoHetero actually ships (§6).
"""
import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import KFold, train_test_split

from autohetero import predictor

try:
    from eval import common
except ImportError:  # run as a script: eval/ itself is on the path
    import common

KINDS = ("forest", "tree")


def sizes(rows):
    """Training rows grouped by size: {(op, elements, batch): {device: (features, seconds)}}."""
    by = defaultdict(dict)
    for f, device, seconds in rows:
        by[f["op"], f["elements"], f["batch"]][device] = (f, seconds)
    return by


def flatten(groups):
    return [(f, device, s) for g in groups for device, (f, s) in g.items()]


def score(model, kind, test):
    """accuracy, regret and time_error of one model kind on test sizes measured on both devices."""
    hits, chosen, oracle, errors = 0, 0.0, 0.0, []
    for measured in test:
        predicted = model.predict(measured["cpu"][0], kind)  # one feature vector per decision, as in dispatch
        actual = {d: s for d, (_, s) in measured.items()}
        pick = min(predicted, key=predicted.get)
        hits += pick == min(actual, key=actual.get)
        chosen, oracle = chosen + actual[pick], oracle + min(actual.values())
        errors += [abs(predicted[d] - actual[d]) / actual[d] for d in actual]
    return {"accuracy": hits / len(test), "regret": chosen / oracle - 1, "time_error": statistics.median(errors), "sizes": len(test)}


def evaluate(train_groups, test_groups):
    model = predictor.Model(flatten(train_groups))
    return {kind: score(model, kind, test_groups) for kind in KINDS}


def run(data, calibration=None):
    """Summary rows and leave-one-op-out rows."""
    groups = sizes(predictor.rows(data))
    keys = sorted(groups)
    testable = [k for k in keys if {"cpu", "gpu"} <= set(groups[k])]
    one_device = [groups[k] for k in keys if k not in testable]  # always training data, never testable
    summary, by_op = [], []

    def add(evaluation, results):
        for kind in KINDS:
            r = results[kind] if isinstance(results, dict) else _mean(kind, results)
            summary.append({"evaluation": evaluation, "model": kind, **r})

    train_keys, test_keys = train_test_split(testable, test_size=0.2, random_state=0)
    add("holdout 80/20", evaluate([groups[k] for k in train_keys] + one_device, [groups[k] for k in test_keys]))
    folds = [
        evaluate([groups[testable[i]] for i in tr] + one_device, [groups[testable[i]] for i in te])
        for tr, te in KFold(5, shuffle=True, random_state=0).split(testable)
    ]
    add("5-fold", folds)
    held_out = []
    for op in sorted({k[0] for k in testable}):
        results = evaluate([groups[k] for k in keys if k[0] != op], [groups[k] for k in testable if k[0] == op])
        held_out.append(results)
        by_op += [{"op": op, "model": kind, **results[kind]} for kind in KINDS]
    add("leave-one-op-out", held_out)
    if calibration:
        model = predictor.Model(predictor.rows(calibration))
        add("calibration -> benchmark", {kind: score(model, kind, [groups[k] for k in testable]) for kind in KINDS})
    return summary, by_op


def _mean(kind, results):
    """Mean of several splits' scores, with the accuracy spread across them."""
    accuracies = [r[kind]["accuracy"] for r in results]
    mean = {m: statistics.mean(r[kind][m] for r in results) for m in ("accuracy", "regret", "time_error")}
    return {**mean, "sizes": sum(r[kind]["sizes"] for r in results), "accuracy_std": statistics.stdev(accuracies)}


def main(data, out_dir, calibration=None):
    summary, by_op = run(data, calibration)
    out_dir.mkdir(parents=True, exist_ok=True)
    common.write_csv(summary, out_dir / "rq1.csv")
    common.write_csv(by_op, out_dir / "rq1_by_op.csv")
    print(f"{'evaluation':26s} {'model':6s} {'accuracy':>9s} {'regret':>8s} {'time err':>9s} {'sizes':>6s}")
    for r in summary:
        spread = f" ±{r['accuracy_std']:.0%}" if "accuracy_std" in r else ""
        print(f"{r['evaluation']:26s} {r['model']:6s} {r['accuracy']:9.1%} {r['regret']:8.1%} {r['time_error']:9.1%} {r['sizes']:6d}{spread}")
    return out_dir / "rq1.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", nargs="?", help="benchmark CSV (default: newest in bench/data)")
    parser.add_argument("--calibration", help="a calibration timings.csv to also score the shipped model")
    parser.add_argument("--out", type=Path, default=common.OUT)
    a = parser.parse_args()
    data = a.data or common.newest("bench/data")
    if not data:
        parser.error("no benchmark data: run bench/collect.py first, or pass a CSV path")
    print("wrote", main(data, a.out, a.calibration))
