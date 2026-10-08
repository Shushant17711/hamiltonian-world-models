"""One trainer for every model (design §6, Req 6.1-6.3).

AdamW + cosine schedule, gradient clipping, a horizon curriculum over the multi-step rollout loss,
early stopping on the validation 32-step normalised MSE, JSONL metrics and resumable checkpoints.

A run directory holds::

    config.yaml       resolved run config (a resume with a different config is refused)
    metrics.jsonl     one record per eval: step, horizon, lr, mean train logs, val_nmse, ...
    ckpt.pt           latest state (model, optimiser, scheduler, sampler RNG, step, best)
    ckpt_best.pt      model weights at the best val_nmse
    normaliser.json   train-split observation statistics
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from hwm.config import Config, config_hash, load_config
from hwm.data.dataset import Batch, Normaliser, WindowDataset
from hwm.models import WorldModel, build
from hwm.utils.seed import seed_everything


class ParamBudgetError(ValueError):
    """The model exceeds the parameter budget (Req 6.3)."""


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_id(cfg: Config) -> str:
    name = cfg.model.name
    if name == "ensemble":
        name = f"{cfg.model.member.name}_ens"
    return f"{cfg.env}-{name}-{cfg.get('obs_mode', 'state')}-s{cfg.seed}"


def horizon_at(step: int, total: int, curriculum: Config) -> int:
    """Horizon in force at ``step``: the last stage whose start fraction has been reached."""
    H = curriculum.horizons[0]
    for h, frac in zip(curriculum.horizons, curriculum.at, strict=True):
        if step >= frac * total:
            H = h
    return H


class Trainer:
    def __init__(self, cfg: Config, run_dir: str | Path, data_dir: str | Path):
        self.cfg, self.tc = cfg, cfg.train
        self.run_dir, self.data_dir = Path(run_dir), Path(data_dir)
        self.device = resolve_device(self.tc.get("device", "auto"))
        seed_everything(cfg.seed)

        self.model: WorldModel = build(cfg).to(self.device)
        n = self.model.n_params()
        # Req 6.3 budgets each model; an ensemble is M models, so its budget applies per member
        members = getattr(self.model, "members", None)
        n_check = max(m.n_params() for m in members) if members is not None else n
        if n_check >= self.tc.get("max_params", 5_000_000):
            raise ParamBudgetError(
                f"{cfg.model.name} has {n_check:,} parameters per model (budget < {self.tc.max_params:,})"
            )
        self.n_params = n

        self.normaliser = Normaliser.from_dir(self.data_dir)
        H_max = max(self.tc.curriculum.horizons)
        kw = {
            "context": self.model.context,
            "obs_mode": self.model.obs_mode,
            "normaliser": self.normaliser,
            "device": self.device,
            "env_name": cfg.env,
        }
        self.train_ds = WindowDataset(self.data_dir, "train", self.tc.curriculum.horizons[0], **kw)
        self._check_horizon(self.train_ds, H_max)
        self.model.prepare(self.normaliser, self.train_ds.obs)
        val_ds = WindowDataset(self.data_dir, "val", self.tc.val_horizon, **kw)
        g = torch.Generator(device=self.device).manual_seed(10_007 + cfg.seed)
        self.val_batch = val_ds.sample(self.tc.val_windows, g)

        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.tc.lr, weight_decay=self.tc.get("weight_decay", 1e-2)
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.tc.steps)
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.step = 0
        self.best = math.inf
        self.bad_evals = 0
        self.done = False

    @staticmethod
    def _check_horizon(ds: WindowDataset, H: int) -> None:
        h0 = ds.horizon
        ds.set_horizon(H)  # raises if trajectories are too short for the final horizon
        ds.set_horizon(h0)

    # --- persistence --------------------------------------------------------------------------
    def _prepare_dir(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = self.run_dir / "config.yaml"
        if cfg_path.exists():
            old = load_config(cfg_path)
            if config_hash(old) != config_hash(self.cfg):
                raise ValueError(f"{self.run_dir} holds a run with a different config; use a new run dir")
        else:
            self.cfg.dump(cfg_path)
        self.normaliser.save(self.run_dir / "normaliser.json")

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "sched": self.sched.state_dict(),
            "gen": self.gen.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "step": self.step,
            "best": self.best,
            "bad_evals": self.bad_evals,
            "done": self.done,
            "config_hash": config_hash(self.cfg),
        }

    def save(self) -> None:
        tmp = self.run_dir / "ckpt.pt.tmp"
        torch.save(self.state_dict(), tmp)
        tmp.replace(self.run_dir / "ckpt.pt")

    def resume(self) -> bool:
        path = self.run_dir / "ckpt.pt"
        if not path.exists():
            return False
        s = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(s["model"])
        self.opt.load_state_dict(s["opt"])
        self.sched.load_state_dict(s["sched"])
        self.gen.set_state(s["gen"].cpu())  # generator states are CPU ByteTensors, even for CUDA generators
        self.step, self.best, self.bad_evals, self.done = s["step"], s["best"], s["bad_evals"], s["done"]
        # models may draw from the global RNG (e.g. PINN collocation points); restore it for exact resume
        torch.set_rng_state(s["torch_rng"].cpu())
        if s["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([r.cpu() for r in s["cuda_rng"]])
        return True

    def _log(self, record: dict[str, Any]) -> None:
        with open(self.run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    # --- training -----------------------------------------------------------------------------
    def _sample_batch(self) -> Batch:
        """One training batch; an ensemble gets one batch per member from its bootstrap resample,
        concatenated along the batch dimension (the ensemble splits it again in ``loss``)."""
        resamples = getattr(self.model, "member_trajs", None)
        if resamples is None:
            return self.train_ds.sample(self.tc.batch, self.gen)
        return Batch.cat([self.train_ds.sample(self.tc.batch, self.gen, idx) for idx in resamples])

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        b, chunks = self.val_batch, []
        for i in range(0, b.ctx.shape[0], 256):
            ro = self.model.rollout(b.ctx[i : i + 256], b.actions[i : i + 256])
            err = (ro.obs[:, 1:].float() - b.target[i : i + 256]).pow(2)
            chunks.append(err.flatten(1).mean(1))
        self.model.train()
        v = torch.cat(chunks).mean().item()
        return v if math.isfinite(v) else math.inf

    def fit(self, stop_after: int | None = None, verbose: bool = False) -> dict[str, Any]:
        """Train to ``train.steps`` (or early stop). ``stop_after`` simulates an interruption."""
        self._prepare_dir()
        self.resume()
        total, H_final = self.tc.steps, max(self.tc.curriculum.horizons)
        sums: dict[str, float] = defaultdict(float)
        n_logged, t0 = 0, time.perf_counter()
        self.model.train()
        while not self.done and self.step < total:
            if stop_after is not None and self.step >= stop_after:
                self.save()
                return {"step": self.step, "best_val_nmse": self.best, "interrupted": True}
            H = horizon_at(self.step, total, self.tc.curriculum)
            if H != self.train_ds.horizon:
                self.train_ds.set_horizon(H)
            batch = self._sample_batch()
            self.opt.zero_grad(set_to_none=True)
            logs = self.model.backward(batch, H)
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.grad_clip)
            if math.isfinite(logs.get("loss", 0.0)) and torch.isfinite(gn):
                self.opt.step()
            else:  # never apply a non-finite update; the step still counts so the schedule is unchanged
                sums["skipped"] += 1.0
                self.opt.zero_grad(set_to_none=True)
                logs = {}
            self.sched.step()
            self.step += 1
            for k, v in logs.items():
                sums[k] += v
            sums["grad_norm"] += gn.item()
            n_logged += 1

            if self.step % self.tc.eval_every == 0 or self.step == total:
                val = self.validate()
                improved = val < self.best
                if improved:
                    self.best = val
                    torch.save(self.model.state_dict(), self.run_dir / "ckpt_best.pt")
                if H == H_final:  # patience only counts once the final horizon is in force
                    self.bad_evals = 0 if improved else self.bad_evals + 1
                    if self.bad_evals >= self.tc.patience:
                        self.done = True
                rec = {
                    "step": self.step,
                    "horizon": H,
                    "lr": self.sched.get_last_lr()[0],
                    **{f"train_{k}": v / n_logged for k, v in sums.items()},
                    "val_nmse": val,
                    "best_val_nmse": self.best,
                    "sec": round(time.perf_counter() - t0, 2),
                }
                self._log(rec)
                if verbose:
                    print(
                        f"step {self.step:>6} H={H:<3} loss={rec.get('train_loss', float('nan')):.4g} "
                        f"val={val:.4g} best={self.best:.4g}",
                        flush=True,
                    )
                sums.clear()
                n_logged = 0
                self.save()
        self.done = True
        self.save()
        return {
            "step": self.step,
            "best_val_nmse": self.best,
            "interrupted": False,
            "n_params": self.n_params,
        }


def load_trained(run_dir: str | Path, best: bool = True, device: str = "auto") -> tuple[WorldModel, Config]:
    """Rebuild a trained model from its run directory (``ckpt_best.pt`` by default)."""
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    dev = resolve_device(device)
    model = build(cfg).to(dev)
    if best and (run_dir / "ckpt_best.pt").exists():
        state = torch.load(run_dir / "ckpt_best.pt", map_location=dev)
    else:
        state = torch.load(run_dir / "ckpt.pt", map_location=dev, weights_only=False)["model"]
    model.load_state_dict(state)
    model.eval()
    return model, cfg
