import numpy as np
import pytest

from hwm.envs import make
from tests.physics_checks import check_dH_finite_difference, relative_drift


@pytest.fixture
def env():
    return make("pendulum")


def test_dH_matches_finite_difference(env):
    qp = env.sample_band(np.random.default_rng(0), (0.05, 0.95), 64)
    check_dH_finite_difference(env, qp)


def test_band_sampling_hits_requested_energies(env):
    qp = env.sample_band(np.random.default_rng(1), (0.7, 0.95), 500)
    e = env.energy(qp) / env.E_ref
    assert qp.shape == (500, 2)
    assert e.min() >= 0.7 - 1e-9 and e.max() <= 0.95 + 1e-9
    assert e.max() - e.min() > 0.2  # actually spread over the band


def test_energy_conserved_over_10k_steps(env):
    qp0 = env.sample_band(np.random.default_rng(2), (0.1, 0.95), 8)
    drift, _ = relative_drift(env, qp0, 10_000)
    assert drift.max() < 1e-6, drift


def test_obs_round_trip(env):
    qp = env.sample_band(np.random.default_rng(3), (0.1, 0.9), 32)
    np.testing.assert_allclose(env.obs_to_qp(env.qp_to_obs(qp)), qp, atol=1e-12)


def test_deterministic_given_seed(env):
    a = env.sample_band(np.random.default_rng(7), (0.1, 0.6), 16)
    b = env.sample_band(np.random.default_rng(7), (0.1, 0.6), 16)
    np.testing.assert_array_equal(a, b)
    u = np.random.default_rng(7).uniform(-2, 2, (16, 1))
    np.testing.assert_array_equal(env.step(a, u), env.step(b, u))


def test_torque_changes_energy_and_is_clipped(env):
    qp = np.zeros((1, 2))
    big = env.step(qp, np.array([[100.0]]))
    capped = env.step(qp, np.array([[env.u_max]]))
    np.testing.assert_allclose(big, capped)
    assert env.energy(capped)[0] > 0
