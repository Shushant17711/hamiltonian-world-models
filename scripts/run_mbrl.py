"""Run the model-based RL loop (H3):

    uv run python scripts/run_mbrl.py --config configs/model/ensemble.yaml env=pendulum obs_mode=pixels

The run config is the model config + ``configs/mbrl.yaml`` under ``mbrl`` (override with
``mbrl.cem.population=200`` etc.). Output: ``<results_root>/mbrl-<env>-<model>-<obs>-s<seed>/``.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hwm.config import load_config
from hwm.planning.mbrl import MBRL, mbrl_run_id


def build_mbrl_config(
    config: str, overrides: list[str], mbrl_config: str | Path = ROOT / "configs" / "mbrl.yaml"
):
    cfg = load_config(config, [o for o in overrides if not o.startswith("mbrl.")])
    m = load_config(mbrl_config).with_overrides(
        [o[len("mbrl.") :] for o in overrides if o.startswith("mbrl.")]
    )
    return cfg.with_overrides({"mbrl": m.to_dict()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = build_mbrl_config(args.config, args.overrides)
    run_dir = (
        Path(args.run_dir) if args.run_dir else Path(cfg.get("results_root", "results")) / mbrl_run_id(cfg)
    )
    print(f"{run_dir}: {cfg.model.name} ({cfg.get('obs_mode', 'state')}) on {cfg.env}", flush=True)
    out = MBRL(cfg, run_dir).run(verbose=True)
    print(f"done: {out}")


if __name__ == "__main__":
    main()
