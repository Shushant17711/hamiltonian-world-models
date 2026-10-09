"""Re-evaluate finished runs with the current evaluation code (adds nmse_curve / drift_curve):
``uv run python scripts/reevaluate.py [results]``. Skips runs whose metrics already have both curves."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate import evaluate_run  # noqa: E402


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    for p in sorted(root.glob("*-state-s*/eval/metrics.json")):
        m = json.loads(p.read_text())
        if "nmse_curve" in m and "drift_curve" in m:
            continue
        t0 = time.perf_counter()
        evaluate_run(p.parent.parent)
        print(f"{p.parent.parent.name}: re-evaluated ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
