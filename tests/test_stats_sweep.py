"""Bootstrap statistics and the sweep runner (Req 7.4)."""

import math
import subprocess
import sys

import numpy as np
import pytest

from hwm.eval.stats import bootstrap_ci, geo_mean_ratio_ci, paired_difference_ci
from scripts.sweep import plan
from tests.test_data_generate import TINY
from tests.test_trainer import FAST, ROOT


def test_ci_covers_true_mean_in_at_least_90_percent_of_trials():
    rng = np.random.default_rng(0)
    hits = 0
    for i in range(200):
        c = bootstrap_ci(rng.normal(3.0, 1.0, size=20), n_resamples=2000, rng_seed=i)
        hits += c.lo <= 3.0 <= c.hi
    assert hits >= 180, hits


def test_ci_is_deterministic_and_ordered():
    v = [0.1, 0.4, 0.35, 0.2]
    a, b = bootstrap_ci(v), bootstrap_ci(v)
    assert a == b and a.lo <= a.mean <= a.hi and a.n == 4


def test_three_seeds_ci_lower_bound_is_the_smallest_value():
    # PREREGISTRATION.md §4: with 3 seeds the 2.5th percentile equals the smallest per-seed value
    assert bootstrap_ci([4.0, 7.0, 9.0]).lo == 4.0


def test_paired_difference_and_ratio():
    a, b = np.array([3.0, 5.0, 4.0]), np.array([1.0, 2.0, 1.5])
    d = paired_difference_ci(a, b)
    assert d.mean == pytest.approx(2.5) and d.lo == pytest.approx(2.0)
    r = geo_mean_ratio_ci(a, b)
    assert r.mean == pytest.approx(np.exp(np.mean(np.log(a / b))))
    assert r.lo == pytest.approx(2.5) and r.hi == pytest.approx(3.0)
    with pytest.raises(ValueError):
        paired_difference_ci([1.0], [1.0, 2.0])


def test_ratio_handles_floor_and_divergence():
    r = geo_mean_ratio_ci([1e-12, 1e-12], [1e-12, 1e-12])  # both floored to 1e-9
    assert r.mean == pytest.approx(1.0)
    r = geo_mean_ratio_ci([math.inf, 1.0, 1.0], [1e-3, 1e-3, 1e-3])  # baseline diverged on one seed
    assert r.mean == math.inf and r.lo == pytest.approx(1000.0)
    r = geo_mean_ratio_ci([1.0, 1.0, 1.0], [math.inf, 1.0, 1.0])  # tested model diverged
    assert r.mean == 0.0 and r.lo == 0.0


@pytest.fixture
def sweep_file(tmp_path):
    p = tmp_path / "s.yaml"
    over = [f"results_root={tmp_path / 'res'}", f"data_root={tmp_path / 'data'}", *FAST, "train.steps=10"]
    over += [f"data.{o}" for o in TINY + ["sizes.train=[16,30]", "sizes.val=[8,30]"]]
    p.write_text(
        "envs: [pendulum, orbit]\nmodels: [mlp, rssm]\nseeds: [0, 1]\noverrides:\n"
        + "".join(f"  - '{o}'\n" for o in over)
    )
    return p


def test_dry_run_lists_expected_ids(sweep_file):
    ids = [r["id"] for r in plan(sweep_file)]
    assert ids == [
        f"{e}-{m}-state-s{s}" for s in (0, 1) for e in ("pendulum", "orbit") for m in ("mlp", "rssm")
    ]
    r = subprocess.run(
        [sys.executable, "scripts/sweep.py", str(sweep_file), "--dry-run"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "8 runs, 8 to do" in r.stdout and "new    orbit-rssm-state-s1" in r.stdout
    r = subprocess.run(
        [sys.executable, "scripts/sweep.py", str(sweep_file), "--dry-run", "--shard", "1/3"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "3 runs, 3 to do" in r.stdout and "pendulum-rssm-state-s0" in r.stdout  # plan[1::3]


def test_sweep_runs_and_skips_finished(sweep_file, tmp_path):
    small = tmp_path / "small.yaml"
    small.write_text(
        sweep_file.read_text().replace("[pendulum, orbit]", "[pendulum]").replace("[0, 1]", "[0]")
    )
    r = subprocess.run(
        [sys.executable, "scripts/sweep.py", str(small)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stderr
    assert all(p["status"] == "done" for p in plan(small))
    assert (tmp_path / "res" / "pendulum-rssm-state-s0" / "eval" / "metrics.json").exists()
    again = subprocess.run(
        [sys.executable, "scripts/sweep.py", str(small)], cwd=ROOT, check=True, capture_output=True, text=True
    )
    assert "2 runs, 0 to do" in again.stdout
