"""Contract tests run on every ground-truth environment (Req 1.1-1.3, 1.5)."""

import numpy as np
import pytest
import torch

from hwm.envs import make, names
from tests.physics_checks import check_dH_finite_difference, relative_drift

# (train band, Req 1.3 drift threshold)
CASES = {
    "pendulum": ((0.10, 0.60), 1e-6),
    "orbit": ((0.8, 1.2), 1e-6),
    "cartpole": ((0.10, 0.60), 1e-5),
    "acrobot": ((0.05, 0.40), 1e-5),
}
# bands that stress the integrator most (closest to separatrix / most chaotic / most eccentric)
HARD_BAND = {"pendulum": (0.7, 0.95), "orbit": (0.8, 1.8), "cartpole": (0.7, 0.95), "acrobot": (0.5, 0.8)}


def test_all_envs_registered():
    assert names() == sorted(CASES)


@pytest.fixture(params=sorted(CASES))
def env(request):
    return make(request.param)


def test_dH_matches_finite_difference(env):
    check_dH_finite_difference(env, env.sample_band(np.random.default_rng(0), HARD_BAND[env.name], 64))


def test_band_sampling_hits_requested_energies(env):
    if env.name == "orbit":
        pytest.skip("orbit bands are over the semi-major axis; see test_envs_orbit")
    lo, hi = (0.5, 0.8) if env.name == "acrobot" else (0.7, 0.95)
    qp = env.sample_band(np.random.default_rng(1), (lo, hi), 400)
    e = env.energy(qp) / env.E_ref
    assert qp.shape == (400, 2 * env.n)
    assert e.min() >= lo - 1e-9 and e.max() <= hi + 1e-9
    assert e.max() - e.min() > 0.5 * (hi - lo)


def _drift_check(env, steps):
    qp0 = env.sample_band(np.random.default_rng(2), HARD_BAND[env.name], 8)
    drift, _ = relative_drift(env, qp0, steps)
    assert drift.max() < CASES[env.name][1], drift


def test_energy_conserved_2k_steps(env):
    _drift_check(env, 2_000)


@pytest.mark.slow
def test_energy_conserved_10k_steps(env):
    _drift_check(env, 10_000)


def test_obs_round_trip_numpy_and_torch(env):
    qp = env.sample_band(np.random.default_rng(3), CASES[env.name][0], 32)
    np.testing.assert_allclose(env.obs_to_qp(env.qp_to_obs(qp)), qp, atol=1e-12)
    t = torch.as_tensor(qp)
    np.testing.assert_allclose(env.qp_to_obs(t).numpy(), env.qp_to_obs(qp), atol=1e-12)
    np.testing.assert_allclose(env.obs_to_qp(env.qp_to_obs(t)).numpy(), qp, atol=1e-12)
    # (..., 2n) shapes are accepted
    assert env.qp_to_obs(qp.reshape(4, 8, -1)).shape == (4, 8, 2 * env.n)


def test_accel_matches_simulator(env):
    """accel(q, v, u) agrees with the finite-difference velocity change of the simulator (and torch == numpy)."""
    rng = np.random.default_rng(4)
    qp = env.sample_band(rng, CASES[env.name][0], 16)
    u = rng.uniform(-env.u_max, env.u_max, (16, env.d_u))
    obs = env.qp_to_obs(qp)
    q, v = obs[:, : env.n], obs[:, env.n :]
    a = env.accel(q, v, u)
    h = 1e-4
    saved = env.dt
    env.dt = h
    try:
        v_next = env.qp_to_obs(env.step(qp, u))[:, env.n :]
        v_prev = env.qp_to_obs(_step_back(env, qp, u))[:, env.n :]
    finally:
        env.dt = saved
    np.testing.assert_allclose(a, (v_next - v_prev) / (2 * h), rtol=1e-5, atol=1e-6)
    a_t = env.accel(torch.as_tensor(q), torch.as_tensor(v), torch.as_tensor(u))
    np.testing.assert_allclose(a_t.numpy(), a, rtol=1e-10, atol=1e-12)


def _step_back(env, qp, u):
    env.dt = -env.dt
    try:
        return env.step(qp, u)
    finally:
        env.dt = -env.dt


def test_energy_is_torch_compatible(env):
    qp = env.sample_band(np.random.default_rng(5), CASES[env.name][0], 8)
    q, p = qp[:, : env.n], qp[:, env.n :]
    H_t = env.H(torch.as_tensor(q), torch.as_tensor(p))
    np.testing.assert_allclose(H_t.numpy(), env.H(q, p), rtol=1e-12)


def test_deterministic_given_seed(env):
    band = CASES[env.name][0]
    a = env.sample_band(np.random.default_rng(7), band, 16)
    b = env.sample_band(np.random.default_rng(7), band, 16)
    np.testing.assert_array_equal(a, b)
    u = np.random.default_rng(7).uniform(-env.u_max, env.u_max, (16, env.d_u))
    np.testing.assert_array_equal(env.step(a, u), env.step(b, u))


def test_actions_are_clipped(env):
    qp = env.sample_band(np.random.default_rng(8), CASES[env.name][0], 2)
    big = env.step(qp, np.full((2, env.d_u), 1e3))
    capped = env.step(qp, np.full((2, env.d_u), env.u_max))
    np.testing.assert_array_equal(big, capped)
