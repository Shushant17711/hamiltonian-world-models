"""Evaluate trained runs: ``uv run python scripts/evaluate.py results/<run> [results/<run> ...]``.

Writes ``<run>/eval/metrics.json`` (design §7, Req 7.1-7.3) with, for the best checkpoint:

* ``nmse[split][h]`` / ``diverged[split][h]``: rollout error at h in ROLLOUT_HORIZONS on test_in and
  test_ood, and on the passive long trajectories split by band (``test_long`` = train band,
  ``test_long_ood``); horizons that do not fit a split are skipped (h = 1000 only fits test_long).
* ``drift[band]``: true-energy drift over DRIFT_STEPS passive steps from the first DRIFT_INITIAL_STATES
  long-trajectory initial states of each band (``test_in`` = train band, ``test_ood``), plus
  ``learned_drift[band]`` for models with an energy head.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hwm.data.dataset import Normaliser
from hwm.data.generate import generate, load_split
from hwm.envs import make
from hwm.eval import thresholds as T
from hwm.eval.energy import energy_drift
from hwm.eval.rollout import feature_var, rollout_nmse
from hwm.train.trainer import load_trained


def _subset(d: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[mask] for k, v in d.items()}


def evaluate_run(run_dir: str | Path, device: str = "auto", drift_steps: int = T.DRIFT_STEPS) -> dict:
    run_dir = Path(run_dir)
    model, cfg = load_trained(run_dir, best=True, device=device)
    env = make(cfg.env)
    data_dir = generate(cfg.data, cfg.get("data_root", "data"))
    norm = Normaliser.load(run_dir / "normaliser.json")
    var = feature_var(env, load_split(data_dir, "train")["obs"])

    long = load_split(data_dir, "test_long")
    splits = {
        "test_in": load_split(data_dir, "test_in"),
        "test_ood": load_split(data_dir, "test_ood"),
        "test_long": _subset(long, ~long["ood"]),
        "test_long_ood": _subset(long, long["ood"]),
    }
    t0 = time.perf_counter()
    out: dict = {
        "run": run_dir.name,
        "env": cfg.env,
        "model": cfg.model.name,
        "obs_mode": cfg.get("obs_mode", "state"),
        "seed": cfg.seed,
        "n_params": model.n_params(),
        "nmse": {},
        "diverged": {},
        "drift": {},
        "learned_drift": {},
        "drift_steps": drift_steps,
    }
    for name, d in splits.items():
        r = rollout_nmse(model, norm, env, d, T.ROLLOUT_HORIZONS, var)
        out["nmse"][name] = {str(h): v["nmse"] for h, v in r.items()}
        out["diverged"][name] = {str(h): v["diverged"] for h, v in r.items()}
    for band, d in (("test_in", splits["test_long"]), ("test_ood", splits["test_long_ood"])):
        x0 = d["obs"][: T.DRIFT_INITIAL_STATES, 0]
        r = energy_drift(model, norm, env, x0, steps=drift_steps)
        out["drift"][band] = r["true"]
        out["learned_drift"][band] = r["learned"]
    out["eval_sec"] = round(time.perf_counter() - t0, 1)
    (run_dir / "eval").mkdir(exist_ok=True)
    (run_dir / "eval" / "metrics.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--drift-steps", type=int, default=T.DRIFT_STEPS)
    args = ap.parse_args()
    for run in args.runs:
        r = evaluate_run(run, args.device, args.drift_steps)
        nm = {s: {h: f"{v:.3g}" for h, v in hv.items()} for s, hv in r["nmse"].items()}
        print(
            f"{run}: nmse {nm}  drift in {r['drift']['test_in']['median']:.3g} "
            f"ood {r['drift']['test_ood']['median']:.3g}  ({r['eval_sec']}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
