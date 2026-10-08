"""Bootstrap statistics over seeds (design §7, Req 7.4; PREREGISTRATION.md §4).

Percentile bootstrap of the mean: BOOTSTRAP_N resamples, alpha = BOOTSTRAP_ALPHA, RNG seeded with
BOOTSTRAP_RNG_SEED. Ratios are bootstrapped in log space over paired per-seed log ratios (and reported
back as geometric-mean ratios); differences over paired per-seed differences.

The interval ends use the "lower"/"higher" percentile (no interpolation between resampled means): it
is marginally conservative and stays well defined when a value is infinite (a diverged run).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from hwm.eval.thresholds import BOOTSTRAP_ALPHA, BOOTSTRAP_N, BOOTSTRAP_RNG_SEED, DRIFT_FLOOR


@dataclass(frozen=True)
class CI:
    mean: float
    lo: float
    hi: float
    n: int

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.mean:.3g} [{self.lo:.3g}, {self.hi:.3g}]"


def _mean(x: np.ndarray, axis=None) -> np.ndarray:
    with np.errstate(invalid="ignore"):  # inf + -inf -> nan is the honest answer there
        return x.mean(axis=axis)


def bootstrap_ci(
    values,
    n_resamples: int = BOOTSTRAP_N,
    alpha: float = BOOTSTRAP_ALPHA,
    rng_seed: int = BOOTSTRAP_RNG_SEED,
) -> CI:
    """Percentile bootstrap CI for the mean of ``values`` (one value per seed)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if len(v) == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    rng = np.random.default_rng(rng_seed)
    means = _mean(v[rng.integers(0, len(v), size=(n_resamples, len(v)))], axis=1)
    lo = np.quantile(means, alpha / 2, method="lower")
    hi = np.quantile(means, 1 - alpha / 2, method="higher")
    return CI(float(_mean(v)), float(lo), float(hi), len(v))


def paired_difference_ci(a, b, **kw) -> CI:
    """CI of mean_s (a_s - b_s) over seeds paired by index."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired values need equal shapes, got {a.shape} and {b.shape}")
    with np.errstate(invalid="ignore"):
        return bootstrap_ci(a - b, **kw)


def geo_mean_ratio_ci(num, den, floor: float = DRIFT_FLOOR, **kw) -> CI:
    """Geometric-mean ratio num/den over paired seeds, both floored at ``floor`` first."""
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    if num.shape != den.shape:
        raise ValueError(f"paired values need equal shapes, got {num.shape} and {den.shape}")
    with np.errstate(invalid="ignore", divide="ignore"):
        logs = np.log(np.maximum(num, floor)) - np.log(np.maximum(den, floor))
    c = bootstrap_ci(logs, **kw)
    return CI(_exp(c.mean), _exp(c.lo), _exp(c.hi), c.n)


def _exp(x: float) -> float:
    return math.exp(x) if math.isfinite(x) else (math.inf if x > 0 else 0.0 if x < 0 else math.nan)
