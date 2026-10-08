"""Estimate the maximal Lyapunov exponent per energy band:
``uv run python scripts/lyapunov.py [--env acrobot] [--steps 4000]`` -> ``results/lyapunov_<env>.json``."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hwm.config import load_config
from hwm.envs import make
from hwm.eval.lyapunov import lyapunov_by_band
from hwm.eval.thresholds import H4_ENV


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default=H4_ENV)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    dcfg = load_config(ROOT / "configs" / "data" / f"{args.env}.yaml")
    bands = {"test_in": tuple(dcfg.bands.train), "test_ood": tuple(dcfg.bands.ood)}
    res = lyapunov_by_band(make(args.env), bands, steps=args.steps)
    out = Path(args.out) / f"lyapunov_{args.env}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"env": args.env, "dt": make(args.env).dt, "bands": res}, indent=1))
    for b, r in res.items():
        print(f"{args.env} {b} {r['band']}: lambda = {r['lambda']:.3f} /s, t_lambda = {r['t_lambda']:.2f} s")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
