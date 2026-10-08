"""Write ``results/verdicts.md`` from evaluated runs: ``uv run python scripts/verdicts.py [results]``."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hwm.eval.hypotheses import write_verdicts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="?", default="results")
    args = ap.parse_args()
    for v in write_verdicts(args.results):
        print(f"{v.hypothesis}: {v.status}  ({', '.join(f'{e.env}={e.status}' for e in v.envs)})")
    print(f"wrote {Path(args.results) / 'verdicts.md'}")


if __name__ == "__main__":
    main()
