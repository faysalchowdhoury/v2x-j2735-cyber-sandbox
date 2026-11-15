"""Command line: ``python -m src.cli run|report``."""

from __future__ import annotations

import argparse

from . import report
from .experiment import run


def main() -> None:
    ap = argparse.ArgumentParser(prog="v2x-sandbox")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="train and evaluate the detector over several seeds")
    r.add_argument("--data", default="data/raw")
    r.add_argument("--out", default="results")
    r.add_argument("--intersection", default="6311")
    r.add_argument("--date", default="2025-02-07")
    r.add_argument("--seeds", type=int, default=5)
    r.add_argument("--start-s", type=float, default=7 * 3600, help="window start, seconds into the day (UTC)")
    r.add_argument("--hours", type=float, default=10.0, help="window length in hours")
    rp = sub.add_parser("report", help="aggregate results into RESULTS.md")
    rp.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.cmd == "run":
        out = run(a.data, a.out, a.intersection, a.date, a.seeds, a.start_s, a.hours)
        report.write(out)
        print(f"wrote {out / 'RESULTS.md'}")
    elif a.cmd == "report":
        print(f"wrote {report.write(a.out)}")


if __name__ == "__main__":
    main()
