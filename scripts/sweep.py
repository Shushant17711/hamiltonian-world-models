"""Run a grid of trainings + evaluations: ``uv run python scripts/sweep.py configs/sweeps/<name>.yaml``.

A sweep file lists::

    envs: [pendulum, cartpole]
    models: [mlp, pinn]          # configs/model/<name>.yaml
    seeds: [0, 1, 2]
    overrides: []                # key=value applied to every run (CLI overrides are appended)

Runs go to ``<results_root>/<env>-<model>-<obs>-s<seed>/``. A run whose ``eval/metrics.json`` exists is
skipped; a run with a checkpoint resumes; a run directory holding a different config is refused by
the trainer. ``--dry-run`` lists every run id with its status and does nothing else. ``--shard i/n``
runs every n-th run of the plan starting at i, so n workers can share one GPU (the models are
kernel-launch bound, so parallel workers overlap well).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # `scripts.*` when run as a script

from hwm.config import load_config
from hwm.data.generate import generate
from hwm.train.trainer import Trainer, run_id
from scripts.evaluate import evaluate_run
from scripts.train import ROOT, build_run_config


def plan(sweep_path: str | Path, overrides: list[str] | None = None) -> list[dict]:
    """Every run of the sweep, in execution order (seed, then env, then model): all of seed 0 first,
    so a partial sweep already gives complete per-seed comparisons."""
    sw = load_config(sweep_path)
    extra = list(sw.get("overrides", ())) + list(overrides or ())
    runs = []
    for seed in sw.seeds:
        for env in sw.envs:
            for model in sw.models:
                cfg = build_run_config(
                    str(ROOT / "configs" / "model" / f"{model}.yaml"), [f"env={env}", f"seed={seed}", *extra]
                )
                rid = run_id(cfg)
                run_dir = Path(cfg.results_root) / rid
                if (run_dir / "eval" / "metrics.json").exists():
                    status = "done"
                elif (run_dir / "ckpt.pt").exists():
                    status = "resume"
                else:
                    status = "new"
                runs.append({"id": rid, "cfg": cfg, "run_dir": run_dir, "status": status})
    ids = [r["id"] for r in runs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"sweep {sweep_path} produces duplicate run ids")
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shard", default="0/1", help="i/n: run only every n-th run of the plan, from i")
    ap.add_argument("overrides", nargs="*", help="key=value overrides for every run")
    args = ap.parse_args()

    i, n = (int(x) for x in args.shard.split("/"))
    if not 0 <= i < n:
        raise SystemExit(f"--shard {args.shard}: need 0 <= i < n")
    runs = plan(args.sweep, args.overrides)[i::n]
    todo = [r for r in runs if r["status"] != "done"]
    print(f"{args.sweep}: {len(runs)} runs, {len(todo)} to do", flush=True)
    for r in runs:
        print(f"  {r['status']:<6} {r['id']}", flush=True)
    if args.dry_run:
        return
    for i, r in enumerate(todo, 1):
        t0 = time.perf_counter()
        cfg = r["cfg"]
        data_dir = generate(cfg.data, cfg.data_root)
        out = Trainer(cfg, r["run_dir"], data_dir).fit()
        m = evaluate_run(r["run_dir"], device=cfg.train.get("device", "auto"))
        print(
            f"[{i}/{len(todo)}] {r['id']}: step {out['step']}, val {out['best_val_nmse']:.4g}, "
            f"drift(in) {m['drift']['test_in']['median']:.3g}  ({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
