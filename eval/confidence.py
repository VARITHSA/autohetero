"""Confidence from tree disagreement, and measuring only when unsure (CARRYOVER.md row 5, offline).

    python eval/confidence.py                   # newest bench/data/*.csv -> eval/out/confidence.csv, confidence_sweep.csv

For every held-out size, each CPU tree is paired with each GPU tree; confidence is the share of the pairs that
agree with the forests' pick (0.5 = a coin toss, 1 = every pair agrees). The sweep then plays the M.Tech rule on
AutoHetero's data: below a threshold, measure both devices and take the winner; above it, trust the prediction.
Probing is charged at zero here, so regret is a lower bound on what selective calibration would cost.
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from autohetero import features, predictor

try:
    from eval import common, rq1
except ImportError:  # run as a script: eval/ itself is on the path
    import common
    import rq1

THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)


def judge(model, test):
    """One row per test size: forest pick, confidence, measured times."""
    out = []
    for measured in test:
        f = measured["cpu"][0]
        x = [features.vector(f)]
        trees = {d: np.array([t.predict(x)[0] for t in m.estimators_]) for d, m in model.forests.items()}
        predicted = model.predict(f)
        pick = min(predicted, key=predicted.get)
        cpu_wins = (trees["cpu"][:, None] < trees["gpu"][None, :]).mean()  # log(s/element): same elements, same order
        actual = {d: s for d, (_, s) in measured.items()}
        out.append({"op": f["op"], "elements": f["elements"], "pick": pick, "best": min(actual, key=actual.get),
                    "confidence": cpu_wins if pick == "cpu" else 1 - cpu_wins, "cpu_s": actual["cpu"], "gpu_s": actual["gpu"]})
    return out


def run(data):
    groups = rq1.sizes(predictor.rows(data))
    keys = sorted(groups)
    testable = [k for k in keys if {"cpu", "gpu"} <= set(groups[k])]
    one_device = [groups[k] for k in keys if k not in testable]
    rows = []
    for op in sorted({k[0] for k in testable}):
        model = predictor.Model(rq1.flatten([groups[k] for k in keys if k[0] != op]))
        rows += [{"evaluation": "leave-one-op-out", **r} for r in judge(model, [groups[k] for k in testable if k[0] == op])]
    for tr, te in KFold(5, shuffle=True, random_state=0).split(testable):
        model = predictor.Model(rq1.flatten([groups[testable[i]] for i in tr] + one_device))
        rows += [{"evaluation": "5-fold", **r} for r in judge(model, [groups[testable[i]] for i in te])]
    return rows, sweep(rows)


def sweep(rows):
    """Accuracy and regret when every size below each threshold is measured instead of predicted."""
    out = []
    for evaluation in sorted({r["evaluation"] for r in rows}):
        part = [r for r in rows if r["evaluation"] == evaluation]
        oracle = sum(min(r["cpu_s"], r["gpu_s"]) for r in part)
        for tau in THRESHOLDS:
            measure = [r["confidence"] < tau for r in part]
            picks = [r["best"] if m else r["pick"] for r, m in zip(part, measure)]
            chosen = sum(r[f"{p}_s"] for r, p in zip(part, picks))
            out.append({"evaluation": evaluation, "threshold": tau, "measured_share": float(np.mean(measure)),
                        "accuracy": float(np.mean([p == r["best"] for r, p in zip(part, picks)])), "regret": chosen / oracle - 1})
    return out


def main(data, out_dir):
    rows, swept = run(data)
    out_dir.mkdir(parents=True, exist_ok=True)
    common.write_csv(rows, out_dir / "confidence.csv")
    common.write_csv(swept, out_dir / "confidence_sweep.csv")
    for evaluation in sorted({r["evaluation"] for r in rows}):
        part = [r for r in rows if r["evaluation"] == evaluation]
        right = [r["confidence"] for r in part if r["pick"] == r["best"]]
        wrong = [r["confidence"] for r in part if r["pick"] != r["best"]]
        print(f"{evaluation}: {len(part)} sizes, median confidence right {np.median(right):.2f}, wrong {np.median(wrong):.2f}")
    print(f"{'evaluation':18s} {'tau':>5s} {'measured':>9s} {'accuracy':>9s} {'regret':>8s}")
    for r in swept:
        print(f"{r['evaluation']:18s} {r['threshold']:5.2f} {r['measured_share']:9.1%} {r['accuracy']:9.1%} {r['regret']:8.1%}")
    return out_dir / "confidence.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", nargs="?", help="benchmark CSV (default: newest in bench/data)")
    parser.add_argument("--out", type=Path, default=common.OUT)
    a = parser.parse_args()
    data = a.data or common.newest("bench/data")
    if not data:
        parser.error("no benchmark data: run bench/collect.py first, or pass a CSV path")
    print("wrote", main(data, a.out))
