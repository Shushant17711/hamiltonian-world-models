"""Train one model: ``uv run python scripts/train.py --config configs/model/mlp.yaml env=cartpole seed=1``.

The run config is the model config (which extends ``configs/train.yaml``) plus ``key=value``
overrides. The dataset config ``configs/data/<env>.yaml`` is folded into the run config under
``data`` (override it with ``data.sizes.train=[...]`` etc.), so the run's ``config.yaml`` pins the
dataset hash. Data is generated if missing. Output: ``<results_root>/<env>-<model>-<obs>-s<seed>/``.
"""

import argparse
from pathlib import Path

from hwm.config import load_config
from hwm.data.generate import generate
from hwm.train.trainer import Trainer, run_id

ROOT = Path(__file__).resolve().parents[1]


def build_run_config(config: str, overrides: list[str]):
    cfg = load_config(config, [o for o in overrides if not o.startswith("data.")])
    data_cfg = load_config(ROOT / "configs" / "data" / f"{cfg.env}.yaml")
    data_cfg = data_cfg.with_overrides([o[len("data.") :] for o in overrides if o.startswith("data.")])
    return cfg.with_overrides({"data": data_cfg.to_dict()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", help="default: <results_root>/<run_id>")
    ap.add_argument("overrides", nargs="*", help="key=value overrides")
    args = ap.parse_args()

    cfg = build_run_config(args.config, args.overrides)
    data_dir = generate(cfg.data, cfg.data_root)
    run_dir = Path(args.run_dir) if args.run_dir else Path(cfg.results_root) / run_id(cfg)
    trainer = Trainer(cfg, run_dir, data_dir)
    print(
        f"{run_dir}: {cfg.model.name}, {trainer.n_params:,} params, device {trainer.device}, data {data_dir}"
    )
    out = trainer.fit(verbose=True)
    print(f"done at step {out['step']}, best val nMSE {out['best_val_nmse']:.4g}")


if __name__ == "__main__":
    main()
