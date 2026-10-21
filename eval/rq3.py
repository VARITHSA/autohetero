"""RQ3 (DESIGN.md §1): zero configuration, one-line integration, low overhead, as one table.

    python eval/overhead.py && python eval/rq3.py     # -> eval/out/rq3.csv

Rows: dispatch overhead per call (from eval/out/overhead.csv), the one-off calibration time on this machine (the
only "configuration" a user waits for), and the AutoHetero lines the demo app adds. The < 1 ms target is a
hypothesis reported against, not asserted.
"""
import argparse
import csv
import json
from pathlib import Path

from autohetero import calibrate

try:
    from eval import common
except ImportError:  # run as a script: eval/ itself is on the path
    import common


def lines_added(app):
    """AutoHetero lines in an app beyond its use of autohetero.ops (its op library)."""
    return [line for line in Path(app).read_text().splitlines() if "autohetero." in line and "ops." not in line]


def main(out_dir, calibration=None):
    rows = []
    overhead = out_dir / "overhead.csv"
    if overhead.exists():
        with open(overhead) as f:
            rows += [{"metric": f"dispatch overhead: {r['case']}", "value": float(r["microseconds_per_call"]), "unit": "us/call"}
                     for r in csv.DictReader(f)]
    else:
        print("no eval/out/overhead.csv: run eval/overhead.py first for the overhead rows")
    cal = Path(calibration) if calibration else calibrate.folder() / "calibration.json"
    seconds = json.loads(cal.read_text())["seconds"] if cal.exists() else "not calibrated yet"
    rows.append({"metric": "calibration, once per machine", "value": seconds, "unit": "s"})
    rows.append({"metric": "lines added to the demo app", "value": len(lines_added(common.ROOT / "demo" / "pipeline.py")), "unit": "lines"})
    out_dir.mkdir(parents=True, exist_ok=True)
    common.write_csv(rows, out_dir / "rq3.csv")
    for r in rows:
        print(f"  {r['metric']:55s} {r['value']!s:>12s} {r['unit']}")
    return out_dir / "rq3.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", help="a calibration.json (default: this machine's)")
    parser.add_argument("--out", type=Path, default=common.OUT)
    a = parser.parse_args()
    print("wrote", main(a.out, a.calibration))
