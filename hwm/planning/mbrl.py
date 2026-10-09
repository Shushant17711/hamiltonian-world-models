"""Model-based RL loop (design §8, Req 8.3; PREREGISTRATION.md §5 H3).

    collect ``random_episodes`` OU-random episodes from the task start
    repeat:
        train ``train_steps_per_iter`` gradient steps on every episode collected so far
        evaluate ``eval_episodes`` MPC episodes at every checkpoint the env-step count has reached
        collect 1 MPC episode (it counts towards the env steps; evaluation episodes do not)
    until ``max_env_steps``

All numbers live in the run config's ``mbrl`` section; the defaults are the pre-registered ones.
Evaluation episodes run in lock-step with a batched CEM, so the 10 of them cost about as much as one.
``stop_at_success`` ends a run at the first checkpoint whose success rate reaches the H3 threshold:
later checkpoints cannot change N80, so the verdict is unaffected (the success curve then ends there).

The run directory holds ``mbrl.jsonl`` (one record per evaluated checkpoint), ``train.jsonl``,
``buffer/train.npz`` (every collected episode) and ``mbrl_state.pt``; a run resumes from its state.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hwm.config import Config, config_hash, load_config
from hwm.data.dataset import Batch, Normaliser, WindowDataset
from hwm.data.generate import ou_actions
from hwm.envs import make as make_env
from hwm.envs.render import render
from hwm.models import build
from hwm.planning.cem import CEMConfig
from hwm.planning.mpc import MPC
from hwm.train.trainer import resolve_device
from hwm.utils.seed import seed_everything

DEFAULTS: dict[str, Any] = {
    "random_episodes": 5,
    "train_steps_per_iter": 2000,
    "train_horizons": [4, 8, 16, 32],  # iteration i trains at horizons[min(i, len - 1)]
    "batch": 128,
    "lr": 3.0e-4,
    "grad_clip": 10.0,
    "eval_at": [1000, 2000, 5000, 10000, 20000, 50000],
    "eval_episodes": 10,
    "max_env_steps": 50000,
    "success_threshold": 0.8,
    "stop_at_success": True,
    "beta": 1.0,  # disagreement penalty for ensembles (ignored for single models)
    "ou_theta": 0.15,
    "ou_sigma": 0.3,
    "cem": {"horizon": 30, "population": 400, "elites": 40, "iterations": 5, "momentum": 0.1},
}


def mbrl_config(cfg: Config) -> dict[str, Any]:
    out = {**DEFAULTS, **(cfg.get("mbrl") or Config()).to_dict()}
    out["cem"] = {**DEFAULTS["cem"], **out.get("cem", {})}
    return out


def mbrl_run_id(cfg: Config) -> str:
    name = cfg.model.name if cfg.model.name != "ensemble" else f"{cfg.model.member.name}_ens"
    return f"mbrl-{cfg.env}-{name}-{cfg.get('obs_mode', 'state')}-s{cfg.seed}"


class MBRL:
    def __init__(self, cfg: Config, run_dir: str | Path):
        self.cfg, self.mc = cfg, mbrl_config(cfg)
        self.run_dir = Path(run_dir)
        self.device = resolve_device(cfg.get("train", Config()).get("device", "auto"))
        self.rng = seed_everything(cfg.seed)
        self.env = make_env(cfg.env)
        dcfg = load_config(Path(__file__).resolve().parents[2] / "configs" / "data" / f"{cfg.env}.yaml")
        c = dcfg.actions.get("centering")
        self.centering = tuple(c) if c else None
        self.model = build(cfg).to(self.device)
        self.is_ens = hasattr(self.model, "members")
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.mc["lr"])
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.plan_gen = torch.Generator(device=self.device).manual_seed(cfg.seed + 1)
        self.obs: list[np.ndarray] = []  # episodes (T+1, d_obs), raw
        self.act: list[np.ndarray] = []  # (T, d_u), raw
        self.normaliser: Normaliser | None = None
        self.env_steps = self.grad_steps = self.iteration = 0
        self.evaluated: list[int] = []
        self.done = False
        self.t0 = time.perf_counter()

    # --- episodes ---------------------------------------------------------------------------------
    def _context(self, hist: list[np.ndarray]) -> torch.Tensor:
        """(n, k, *obs) model input from per-env observation histories [(t+1, n, d_obs)]."""
        k = self.model.context
        frames = [hist[max(0, len(hist) - k + i)] for i in range(k)]  # pad by repeating the first frame
        x = torch.as_tensor(np.stack(frames, 1), dtype=torch.float32)  # (n, k, d_obs)
        if self.model.obs_mode == "pixels":
            return render(self.env.name, x.to(self.device))
        return self.normaliser.norm(x)

    def run_episodes(self, n: int, policy: str) -> tuple[np.ndarray, np.ndarray]:
        """n episodes in lock-step from the task start: ('random' | 'mpc') -> obs (n, T+1, d), act (n, T, d_u)."""
        env, T = self.env, self.env.episode_len
        qp = env.task_start(n)
        hist = [env.qp_to_obs(qp)]
        acts = []
        if policy == "random":
            seq = ou_actions(self.rng, n, T, env.d_u, env.u_max, self.mc["ou_theta"], self.mc["ou_sigma"])
        else:
            mpc = MPC(
                self.model,
                env,
                self.normaliser,
                CEMConfig(**self.mc["cem"]),
                beta=self.mc["beta"] if self.is_ens else 0.0,
                generator=self.plan_gen,
                n=n,
            )
        for t in range(T):
            if policy == "random":
                u = seq[:, t]
                if (
                    self.centering is not None
                ):  # as in dataset generation (design §3): keeps the cart on the track
                    o = hist[-1]
                    u = u - self.centering[0] * o[:, :1] - self.centering[1] * o[:, env.n : env.n + 1]
            else:
                a, _ = mpc.act(self._context(hist))
                u = a.double().cpu().numpy() * env.u_max
            qp = env.step(qp, u)
            hist.append(env.qp_to_obs(qp))
            acts.append(np.clip(u, -env.u_max, env.u_max))
        return np.stack(hist, 1), np.stack(acts, 1)

    def add(self, obs: np.ndarray, act: np.ndarray) -> None:
        self.obs += list(obs)
        self.act += list(act)
        self.env_steps += act.shape[0] * act.shape[1]
        if self.normaliser is None:  # fixed after the random warm-up (state mode needs it)
            self.normaliser = Normaliser.fit(np.stack(self.obs))
        self._write_buffer()

    def _write_buffer(self) -> None:
        d = self.run_dir / "buffer"
        d.mkdir(parents=True, exist_ok=True)
        obs, act = np.stack(self.obs), np.stack(self.act)
        np.savez(d / "train.npz", obs=obs, act=act, passive=np.zeros(len(obs), dtype=bool))

    # --- training ---------------------------------------------------------------------------------
    def train(self) -> dict[str, float]:
        hs = self.mc["train_horizons"]
        H = min(hs[min(self.iteration, len(hs) - 1)], self.env.episode_len - self.model.context)
        ds = WindowDataset(
            self.run_dir / "buffer",
            "train",
            H,
            context=self.model.context,
            obs_mode=self.model.obs_mode,
            normaliser=self.normaliser,
            device=self.device,
            env_name=self.env.name,
        )
        N = len(self.obs)
        if self.is_ens:  # each member trains on its own bootstrap resample of the episodes so far
            self.model.member_trajs = [
                torch.as_tensor(
                    np.random.default_rng([s, self.iteration]).integers(0, N, N), device=self.device
                )
                for s in self.model.member_seeds
            ]
        self.model.train()
        sums: dict[str, float] = {}
        for _ in range(self.mc["train_steps_per_iter"]):
            if self.is_ens:
                batch = Batch.cat(
                    [ds.sample(self.mc["batch"], self.gen, idx) for idx in self.model.member_trajs]
                )
            else:
                batch = ds.sample(self.mc["batch"], self.gen)
            self.opt.zero_grad(set_to_none=True)
            logs = self.model.backward(batch, H)
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.mc["grad_clip"])
            if math.isfinite(logs.get("loss", 0.0)) and torch.isfinite(gn):
                self.opt.step()
            else:  # never apply a non-finite update (as in the shared trainer, amendment 1)
                self.opt.zero_grad(set_to_none=True)
                logs = {"skipped": 1.0}
            self.grad_steps += 1
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + v
        n = max(self.mc["train_steps_per_iter"], 1)
        rec = {
            "iteration": self.iteration,
            "env_steps": self.env_steps,
            "grad_steps": self.grad_steps,
            "horizon": H,
            **{k: v / n for k, v in sums.items()},
            "sec": self._sec(),
        }
        self._append("train.jsonl", rec)
        return rec

    # --- evaluation -------------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, checkpoint: int) -> dict[str, Any]:
        obs, act = self.run_episodes(self.mc["eval_episodes"], "mpc")
        success = self.env.success(obs)
        ret = self.env.reward(obs[:, 1:], act).sum(-1)
        rec = {
            "checkpoint": checkpoint,
            "env_steps": self.env_steps,
            "success_rate": float(success.mean()),
            "mean_return": float(ret.mean()),
            "returns": [float(r) for r in ret],
            "episodes": len(self.obs),
            "grad_steps": self.grad_steps,
            "sec": self._sec(),
        }
        self._append("mbrl.jsonl", rec)
        return rec

    # --- loop -------------------------------------------------------------------------------------
    def run(self, max_iterations: int | None = None, verbose: bool = False) -> dict[str, Any]:
        self._prepare_dir()
        self.resume()
        if not self.obs:
            o, a = self.run_episodes(self.mc["random_episodes"], "random")
            self.add(o, a)
            self.save()
        its = 0
        while not self.done:
            if max_iterations is not None and its >= max_iterations:
                break
            self.train()
            for c in self.mc["eval_at"]:
                if c <= self.env_steps and c not in self.evaluated:
                    rec = self.evaluate(c)
                    self.evaluated.append(c)
                    if verbose:
                        print(
                            f"[{self.cfg.env}] checkpoint {c} (env steps {self.env_steps}): "
                            f"success {rec['success_rate']:.0%}, return {rec['mean_return']:.1f}, "
                            f"{rec['sec']:.0f}s",
                            flush=True,
                        )
                    if self.mc["stop_at_success"] and rec["success_rate"] >= self.mc["success_threshold"]:
                        self.done = True
            all_evaluated = all(
                c in self.evaluated for c in self.mc["eval_at"] if c <= self.mc["max_env_steps"]
            )
            if self.env_steps >= self.mc["max_env_steps"] or all_evaluated:
                self.done = True
            if not self.done:
                o, a = self.run_episodes(1, "mpc")
                self.add(o, a)
            self.iteration += 1
            its += 1
            self.save()
        return {"env_steps": self.env_steps, "evaluated": self.evaluated, "done": self.done}

    # --- persistence ------------------------------------------------------------------------------
    def _sec(self) -> float:
        return round(time.perf_counter() - self.t0 + getattr(self, "_sec_offset", 0.0), 1)

    def _append(self, name: str, rec: dict) -> None:
        with open(self.run_dir / name, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _prepare_dir(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        p = self.run_dir / "config.yaml"
        if p.exists():
            if config_hash(load_config(p)) != config_hash(self.cfg):
                raise ValueError(f"{self.run_dir} holds an MBRL run with a different config")
        else:
            self.cfg.dump(p)

    def save(self) -> None:
        state = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "gen": self.gen.get_state(),
            "plan_gen": self.plan_gen.get_state(),
            "np_rng": self.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "normaliser": self.normaliser.state_dict() if self.normaliser else None,
            "env_steps": self.env_steps,
            "grad_steps": self.grad_steps,
            "iteration": self.iteration,
            "evaluated": self.evaluated,
            "done": self.done,
            "sec": self._sec(),
        }
        tmp = self.run_dir / "mbrl_state.pt.tmp"
        torch.save(state, tmp)
        tmp.replace(self.run_dir / "mbrl_state.pt")

    def resume(self) -> bool:
        p = self.run_dir / "mbrl_state.pt"
        if not p.exists():
            return False
        s = torch.load(p, map_location=self.device, weights_only=False)
        self.model.load_state_dict(s["model"])
        self.opt.load_state_dict(s["opt"])
        self.gen.set_state(s["gen"].cpu())  # generator states are CPU ByteTensors, even for CUDA
        self.plan_gen.set_state(s["plan_gen"].cpu())
        self.rng.bit_generator.state = s["np_rng"]
        torch.set_rng_state(s["torch_rng"].cpu())
        self.normaliser = Normaliser.from_state_dict(s["normaliser"]) if s["normaliser"] else None
        self.env_steps, self.grad_steps = s["env_steps"], s["grad_steps"]
        self.iteration, self.evaluated, self.done = s["iteration"], list(s["evaluated"]), s["done"]
        self._sec_offset, self.t0 = s["sec"], time.perf_counter()
        with np.load(self.run_dir / "buffer" / "train.npz") as z:
            self.obs, self.act = list(z["obs"]), list(z["act"])
        return True
