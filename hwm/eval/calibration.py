"""Calibration of ensemble disagreement against true error (design §7, Req 8.4).

For every (trajectory, horizon) pair: disagreement d (std over members of the decoded prediction,
averaged over dims) and true error e (squared error of the ensemble mean, averaged over dims). Reports
Spearman's rho between d and e per horizon and pooled, and a 10-bin reliability curve: bins of equal
count over d, with the mean d and the mean sqrt(e) (RMSE) in each bin. A calibrated ensemble has
rho near 1 and a reliability curve near the diagonal.
"""

from __future__ import annotations

import numpy as np
import torch


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank), 0-based."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / counts)[inv]


def spearman(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def reliability(disagreement, sq_error, bins: int = 10) -> dict[str, list[float]]:
    d = np.asarray(disagreement, dtype=np.float64).ravel()
    e = np.asarray(sq_error, dtype=np.float64).ravel()
    ok = np.isfinite(d) & np.isfinite(e)
    d, e = d[ok], e[ok]
    order = np.argsort(d)
    chunks = np.array_split(order, bins)
    return {
        "disagreement": [float(d[c].mean()) for c in chunks if len(c)],
        "rmse": [float(np.sqrt(e[c].mean())) for c in chunks if len(c)],
        "count": [len(c) for c in chunks if len(c)],
    }


@torch.no_grad()
def calibration(ensemble, normaliser, env, data: dict[str, np.ndarray], horizons, chunk: int = 128) -> dict:
    """Calibration of a trained ensemble on a split (state mode; normalised units)."""
    from hwm.eval.rollout import model_device, model_dtype

    dev, dt = model_device(ensemble), model_dtype(ensemble)
    k, T = ensemble.context, data["act"].shape[1]
    hs = [h for h in horizons if k - 1 + h <= T]
    H = max(hs)
    obs = torch.as_tensor(data["obs"], dtype=torch.float32)
    act = torch.as_tensor(data["act"], dtype=torch.float32) / env.u_max
    D, E = [], []
    for i in range(0, obs.shape[0], chunk):
        o = normaliser.norm(obs[i : i + chunk]).to(dev, dt)
        ro = ensemble.rollout(o[:, :k], act[i : i + chunk, k - 1 : k - 1 + H].to(dev, dt))
        idx = torch.as_tensor(hs, device=dev)
        D.append(ro.disagreement[:, idx].cpu())
        err = (ro.obs[:, idx] - o[:, k - 1 + idx]).pow(2).flatten(2).mean(-1)
        E.append(err.cpu())
    D, E = torch.cat(D).double().numpy(), torch.cat(E).double().numpy()
    return {
        "horizons": hs,
        "spearman": {str(h): spearman(D[:, j], E[:, j]) for j, h in enumerate(hs)},
        "spearman_pooled": spearman(D, E),
        "reliability": reliability(D, E),
    }
