"""Trajectory dataset generation with energy-band splits (Req 2.1-2.3, design §3).

Layout: ``<root>/<env>/<config_hash>/{train,val,test_in,test_ood,test_long}.npz`` with arrays
``obs (N, T+1, d_obs)``, ``qp (N, T+1, 2n)``, ``act (N, T, d_u)``, ``energy (N, T+1)``,
``passive (N,)`` and ``ood (N,)``. ``test_long`` holds passive trajectories from both bands
(``ood`` tells them apart). Images are not stored; they are rendered at batch time.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from hwm.config import Config, config_hash, load_config
from hwm.envs import make
from hwm.envs.base import Env

SPLITS = ("train", "val", "test_in", "test_ood", "test_long")
_SPLIT_ID = {name: i for i, name in enumerate(SPLITS)}


def ou_actions(rng: np.random.Generator, N: int, T: int, d_u: int, u_max: float, theta: float, sigma: float):
    """Ornstein-Uhlenbeck actions (N, T, d_u), started from the stationary law, clipped to u_max."""
    s = sigma * u_max
    x = rng.normal(0.0, s / np.sqrt(theta * (2 - theta)), (N, d_u))
    out = np.empty((N, T, d_u))
    for t in range(T):
        out[:, t] = x
        x = x - theta * x + s * rng.standard_normal((N, d_u))
    return np.clip(out, -u_max, u_max)


def rollout(
    env: Env, qp0: np.ndarray, act: np.ndarray, centering: tuple[float, float] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(N, 2n), (N, T, d_u) -> (qp (N, T+1, 2n), applied actions (N, T, d_u)).

    ``centering = (kp, kd)`` adds a PD term on the first coordinate (the cart position) to the
    open-loop actions, so long random-force rollouts stay on the track. The *applied* (clipped)
    actions are returned and stored, so the data stays exactly consistent with the simulator.
    """
    qps, used = [qp0], []
    for t in range(act.shape[1]):
        u = act[:, t]
        if centering is not None:
            obs = env.qp_to_obs(qps[-1])
            u = u - centering[0] * obs[:, :1] - centering[1] * obs[:, env.n : env.n + 1]
        u = np.clip(u, -env.u_max, env.u_max)
        used.append(u)
        qps.append(env.step(qps[-1], u))
    return np.stack(qps, 1), np.stack(used, 1)


def generate_split(
    env: Env,
    rng: np.random.Generator,
    band: tuple[float, float],
    N: int,
    T: int,
    passive_frac: float,
    ou_theta: float,
    ou_sigma: float,
    centering: tuple[float, float] | None = None,
    max_rounds: int = 50,
) -> dict[str, np.ndarray]:
    """N trajectories from ``band``; any trajectory that leaves ``env.valid_state`` is redrawn."""
    n_passive = int(round(passive_frac * N))
    passive = np.zeros(N, dtype=bool)
    passive[:n_passive] = True
    rng.shuffle(passive)
    qp = np.empty((N, T + 1, 2 * env.n))
    act = np.zeros((N, T, env.d_u))
    todo = np.arange(N)
    for _ in range(max_rounds):
        k = len(todo)
        a = ou_actions(rng, k, T, env.d_u, env.u_max, ou_theta, ou_sigma)
        pas = passive[todo]
        a[pas] = 0.0
        traj, a = rollout(env, env.sample_band(rng, band, k), a, centering)
        a[pas] = 0.0  # passive means u = 0 exactly, no centering
        if centering is not None and pas.any():
            traj[pas] = rollout(env, traj[pas, 0], a[pas])[0]
        ok = env.valid_state(traj).all(-1)
        qp[todo[ok]], act[todo[ok]] = traj[ok], a[ok]
        todo = todo[~ok]
        if len(todo) == 0:
            break
    else:  # pragma: no cover
        raise RuntimeError(f"{env.name}: {len(todo)} trajectories still invalid after {max_rounds} rounds")
    return {
        "obs": env.qp_to_obs(qp),
        "qp": qp,
        "act": act,
        "energy": env.energy(qp),
        "passive": passive,
        "ood": np.zeros(N, dtype=bool),
    }


def _split_rng(seed: int, split: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, _SPLIT_ID[split]]))


def build_split(env: Env, cfg: Config, split: str) -> dict[str, np.ndarray]:
    rng = _split_rng(cfg.seed, split)
    N, T = cfg.sizes[split]
    act = cfg.actions
    centering = act.get("centering")
    kw = dict(ou_theta=act.ou_theta, ou_sigma=act.ou_sigma, centering=tuple(centering) if centering else None)
    if split == "test_long":
        parts = []
        for ood, band in ((False, cfg.bands.train), (True, cfg.bands.ood)):
            d = generate_split(env, rng, tuple(band), N, T, passive_frac=1.0, **kw)
            d["ood"][:] = ood
            parts.append(d)
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    band = cfg.bands.ood if split == "test_ood" else cfg.bands.train
    d = generate_split(env, rng, tuple(band), N, T, passive_frac=act.passive_frac, **kw)
    d["ood"][:] = split == "test_ood"
    return d


def dataset_dir(cfg: Config, root: str | Path = "data") -> Path:
    return Path(root) / cfg.env / config_hash(cfg)


def generate(cfg: Config, root: str | Path = "data", force: bool = False) -> Path:
    """Write every split for ``cfg``; skip if the hash directory is already complete."""
    out = dataset_dir(cfg, root)
    if out.is_dir() and not force and all((out / f"{s}.npz").exists() for s in SPLITS):
        return out
    env = make(cfg.env)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for split in SPLITS:
        np.savez(tmp / f"{split}.npz", **build_split(env, cfg, split))
    cfg.dump(tmp / "config.yaml")
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return out


def load_split(path: str | Path, split: str) -> dict[str, np.ndarray]:
    with np.load(Path(path) / f"{split}.npz") as z:
        return {k: z[k] for k in z.files}


def load_data_config(path: str | Path, overrides=None) -> Config:
    return load_config(path, overrides)
