"""H1/H2 evaluator on synthetic result trees (Req 7.5)."""

import json
import math

import pytest

from hwm.eval import thresholds as T
from hwm.eval.hypotheses import MODEL_IDS, write_verdicts

ENVS = ("pendulum", "cartpole", "acrobot", "orbit")


def _write(root, env, letter, seed, drift=1e-2, nmse=1.0, curve=None):
    name = MODEL_IDS[letter]
    d = root / f"{env}-{name}-state-s{seed}" / "eval"
    d.mkdir(parents=True)
    m = {
        "env": env,
        "model": name,
        "obs_mode": "state",
        "seed": seed,
        "nmse": {"test_ood": {"10": 0.1, "100": nmse}},
        "drift": {"test_in": {"median": drift}, "test_ood": {"median": drift}},
    }
    if curve is not None:
        m["nmse_curve"] = {
            "test_long": {str(h): v for h, v in curve.items()},
            "test_in": {str(h): v for h, v in curve.items() if h <= 100},
        }
    d.joinpath("metrics.json").write_text(json.dumps(m))


def _tree(root, drift_e=None, nmse=None, skip=()):
    """drift_e[env] -> per-seed E drift (C is 1e-2); nmse[env][letter] -> per-seed nMSE."""
    for env in ENVS:
        for letter in "ABCDE":
            for s in T.SEEDS:
                if (env, letter, s) in skip:
                    continue
                drift = 1e-2 if letter != "E" else (drift_e or {}).get(env, [1e-4] * 3)[s]
                base = {"E": [0.1] * 3}.get(letter, [0.5, 0.6, 0.4])
                n = (nmse or {}).get(env, {}).get(letter, base)[s]
                _write(root, env, letter, s, drift, n)


def _status(vs, keys=("H1", "H2")):
    return {v.hypothesis: v.status for v in vs if v.hypothesis in keys}


def test_both_supported(tmp_path):
    _tree(tmp_path)
    vs = write_verdicts(tmp_path)
    assert _status(vs) == {"H1": "supported", "H2": "supported"}
    md = (tmp_path / "verdicts.md").read_text()
    assert "| H1 | **supported** |" in md and "B − E (privileged)" in md
    assert json.loads((tmp_path / "verdicts.json").read_text())[0]["status"] == "supported"


def test_h1_fails_when_ratio_too_small_or_one_seed_weak(tmp_path):
    # ratio 5x everywhere -> every env fails the >= 10 rule
    _tree(tmp_path, drift_e={e: [2e-3] * 3 for e in ENVS})
    assert _status(write_verdicts(tmp_path))["H1"] == "not supported"


def test_h1_ci_lower_bound_uses_the_weakest_seed(tmp_path):
    # geometric mean ~ 21x but one seed only 2.5x -> lower bound < 3 -> fail on all three envs
    _tree(tmp_path, drift_e={e: [4e-3, 1e-4, 1e-4] for e in ENVS})
    v = write_verdicts(tmp_path)[0]
    r = v.envs[0].detail["ratio"]
    assert r.mean > 10 and r.lo == pytest.approx(2.5)
    assert v.status == "not supported"


def test_h1_two_of_three_is_enough(tmp_path):
    _tree(tmp_path, drift_e={"orbit": [1e-2] * 3})
    v = write_verdicts(tmp_path)[0]
    assert [e.status for e in v.envs] == ["pass", "pass", "fail"] and v.status == "supported"


def test_h1_divergent_E_fails(tmp_path):
    _tree(tmp_path, drift_e={e: [math.inf, 1e-4, 1e-4] for e in ENVS})
    assert _status(write_verdicts(tmp_path))["H1"] == "not supported"


def test_h2_one_unbeaten_baseline_fails_the_env(tmp_path):
    # PINN beats E on two envs -> only 2 of 4 pass -> not supported
    _tree(tmp_path, nmse={e: {"B": [0.05, 0.05, 0.05]} for e in ("pendulum", "cartpole")})
    v = write_verdicts(tmp_path)[1]
    assert [e.status for e in v.envs] == ["fail", "fail", "pass", "pass"] and v.status == "not supported"
    assert v.envs[0].detail["beaten"] == {"A": True, "B": False, "C": True, "D": True}
    assert "✗" in (tmp_path / "verdicts.md").read_text()


def test_h2_needs_every_seed_to_favour_E(tmp_path):
    _tree(tmp_path, nmse={e: {"D": [0.5, 0.6, 0.09]} for e in ENVS})  # one seed of D beats E
    assert _status(write_verdicts(tmp_path))["H2"] == "not supported"


def test_incomplete_and_exploratory(tmp_path):
    _tree(tmp_path, skip={("orbit", "C", 2), ("pendulum", "A", 0)})
    _write(tmp_path, "pendulum", "E", 7)  # extra seed: exploratory only
    vs = write_verdicts(tmp_path)
    h1, h2, h4 = vs
    assert h4.status == "incomplete"  # no Lyapunov file and no curves
    assert [e.status for e in h1.envs] == ["pass", "pass", "incomplete"] and h1.status == "supported"
    # orbit-C is also an H2 baseline: pendulum and orbit are open, 2 passes are not yet 3
    assert [e.status for e in h2.envs] == ["incomplete", "pass", "pass", "incomplete"]
    assert h2.status == "incomplete"
    md = (tmp_path / "verdicts.md").read_text()
    assert "`orbit-latent_ode-state-s2`" in md and "`pendulum-hamiltonian-state-s7`" in md


def test_undecided_is_incomplete(tmp_path):
    _tree(tmp_path, drift_e={"pendulum": [1e-2] * 3}, skip={("cartpole", "E", 0)})
    assert write_verdicts(tmp_path)[0].status == "incomplete"  # 1 pass, 1 fail, 1 open


def _h4_tree(root, late_ratios):
    """Acrobot curves with t_lambda = 1 s (dt = 0.05): h < 20 is early, h > 60 is late."""
    (root / "lyapunov_acrobot.json").write_text(
        json.dumps({"env": "acrobot", "dt": 0.05, "bands": {"test_in": {"lambda": 1.0, "t_lambda": 1.0}}})
    )
    for s in T.SEEDS:
        e_curve = {h: 0.001 * h for h in T.H4_HORIZONS}
        for letter in "ABCDE":
            if letter == "E":
                curve = e_curve
            else:
                worse = 1.0 if letter == "A" else 1.5  # A is the strongest baseline everywhere
                curve = {
                    h: e_curve[h] * worse * (5.0 if h * 0.05 < 1 else late_ratios[s]) for h in T.H4_HORIZONS
                }
            _write(root, "acrobot", letter, s, curve=curve)


def test_h4_confirmed(tmp_path):
    _h4_tree(tmp_path, late_ratios=[0.8, 1.0, 1.25])
    h4 = write_verdicts(tmp_path)[2]
    d = h4.envs[0].detail
    assert (d["part_a"], d["part_b"], h4.status) == ("holds", "holds", "confirmed")
    assert all(r["baseline"] == "A" for r in d["rows"]) and len(d["rows_actuated"]) == 7
    assert d["rows"][0]["advantage"].mean == pytest.approx(5.0)
    assert "| H4 | **confirmed** |" in (tmp_path / "verdicts.md").read_text()


def test_h4_partially_confirmed_when_advantage_persists(tmp_path):
    _h4_tree(tmp_path, late_ratios=[3.0, 3.0, 3.0])
    h4 = write_verdicts(tmp_path)[2]
    assert (h4.envs[0].detail["part_b"], h4.status) == ("fails", "partially confirmed")
