"""Generate trajectory datasets: ``uv run python scripts/gen_data.py --config configs/data/pendulum.yaml [a.b=c ...]``."""

import argparse
import time

from hwm.config import load_config
from hwm.data.generate import generate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", nargs="+", required=True, help="one or more data config files")
    ap.add_argument("--root", default="data")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("overrides", nargs="*", help="key=value overrides applied to every config")
    args = ap.parse_args()
    for path in args.config:
        cfg = load_config(path, args.overrides)
        t0 = time.perf_counter()
        out = generate(cfg, args.root, force=args.force)
        print(f"{cfg.env}: {out} ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
