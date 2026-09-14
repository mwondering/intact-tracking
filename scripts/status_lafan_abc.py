"""Read-only compact progress report for a LaFAN A/B/C experiment."""

import argparse
import json
import re
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("runs/abc_lafan_20260907_r2"))
    args = parser.parse_args()
    for arm in "ABC":
        path = args.root / f"{arm}_train.log"
        if not path.exists():
            print(arm, "not started")
            continue
        log = path.read_text(errors="replace")
        iterations = re.findall(r"Learning iteration (\d+)/(\d+)", log)
        durations = [float(v) for v in re.findall(r"Iteration time: ([\d.]+)s", log)]
        completed, budget = (int(iterations[-1][0]) + 1, int(iterations[-1][1])) if iterations else (0, 1000)
        recent = sum(durations[-20:]) / max(1, len(durations[-20:]))
        status = {"arm": arm, "completed": completed, "budget": budget,
                  "recent_seconds_per_update": round(recent, 2),
                  "estimated_minutes_remaining": round((budget - completed) * recent / 60, 1),
                  "log_age_seconds": round(time.time() - path.stat().st_mtime, 1),
                  "exception_logged": "Traceback (most recent call last)" in log,
                  "completion_audit": (args.root / arm / "completion_audit.json").exists()}
        print(json.dumps(status))
        if status["exception_logged"]:
            traceback_start = log.rfind("Traceback (most recent call last)")
            print(log[traceback_start:traceback_start + 4000])
    print(json.dumps({"completed_evaluations": len(list((args.root / "eval").glob("*.json"))),
                      "comparison_exists": (args.root / "comparison.json").exists()}))


if __name__ == "__main__":
    main()
