"""Ensemble calibration (Req 8.4)."""

import numpy as np
import pytest
from scipy.stats import spearmanr

from hwm.eval.calibration import rankdata, reliability, spearman


def test_spearman_perfect_and_shuffled():
    rng = np.random.default_rng(0)
    d = rng.random(2000)
    assert spearman(d, d**3) == pytest.approx(1.0)
    assert spearman(d, -d) == pytest.approx(-1.0)
    assert abs(spearman(d, rng.permutation(d))) < 0.1


def test_spearman_matches_scipy_with_ties_and_nans():
    rng = np.random.default_rng(1)
    a, b = rng.integers(0, 5, 300).astype(float), rng.random(300)
    assert spearman(a, b) == pytest.approx(spearmanr(a, b).statistic)
    a[3] = np.nan
    assert np.isfinite(spearman(a, b))
    np.testing.assert_allclose(rankdata(np.array([3.0, 1.0, 3.0])), [1.5, 0.0, 1.5])


def test_reliability_curve_on_calibrated_data():
    rng = np.random.default_rng(2)
    sigma = rng.uniform(0.1, 1.0, 20000)
    err2 = (sigma * rng.standard_normal(20000)) ** 2
    r = reliability(sigma, err2)
    assert len(r["rmse"]) == 10 and sum(r["count"]) == 20000
    np.testing.assert_allclose(r["rmse"], r["disagreement"], rtol=0.08)
    assert r["disagreement"] == sorted(r["disagreement"])
