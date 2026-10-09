"""Run the H3 MBRL grid: ``uv run python scripts/mbrl_sweep.py configs/sweeps/mbrl.yaml [--shard i/n] [--dry-run]``.

Order is seed -> env -> arm; a run whose ``mbrl_state.pt`` says done is skipped, an interrupted one resumes.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from hwm.config import load_config
from hwm.models.registry import get as get_model
from hwm.planning.mbrl import MBRL, mbrl_run_id
from scripts.run_mbrl import build_mbrl_config

for _name in ("mlp", "pinn", "latent_ode", "rssm", "hamiltonian", "ensemble"):
    get_model(_name)


def plan(sweep: str | Path) -> list[dict]:
    sw = load_config(sweep)
    runs = []
    for seed in sw.seeds:
        for env in sw.envs:
            for arm in sw.arms:
                cfg = build_mbrl_config(
                    str(ROOT / "configs" / "model" / f"{arm.config}.yaml"),
                    [f"env={env}", f"seed={seed}", f"obs_mode={arm.obs_mode}"],
                    ROOT / sw.mbrl_config,
                )
                run_dir = Path(cfg.get("results_root", "results")) / mbrl_run_id(cfg)
                state = run_dir / "mbrl_state.pt"
                done = state.exists() and torch.load(state, map_location="cpu", weights_only=False)["done"]
                status = "done" if done else ("resume" if state.exists() else "new")
                runs.append({"cfg": cfg, "run_dir": run_dir, "status": status})
    return runs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    i, n = (int(x) for x in args.shard.split("/"))
    runs = plan(args.sweep)[i::n]
    todo = [r for r in runs if r["status"] != "done"]
    print(f"{args.sweep}: {len(runs)} runs, {len(todo)} to do", flush=True)
    for r in runs:
        print(f"  {r['status']:<6} {r['run_dir'].name}", flush=True)
    if args.dry_run:
        return
    for k, r in enumerate(todo, 1):
        t0 = time.perf_counter()
        out = MBRL(r["cfg"], r["run_dir"]).run(verbose=True)
        print(f"[{k}/{len(todo)}] {r['run_dir'].name}: {out}  ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
