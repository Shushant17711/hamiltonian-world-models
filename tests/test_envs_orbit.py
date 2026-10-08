import numpy as np
import pytest

from hwm.envs import make
from tests.physics_checks import check_dH_finite_difference, relative_drift


@pytest.fixture
def env():
    return make("orbit")


def test_dH_matches_finite_difference(env):
    check_dH_finite_difference(env, env.sample_band(np.random.default_rng(0), (0.8, 1.8), 64))


def test_band_sampling_recovers_elements(env):
    rng = np.random.default_rng(1)
    qp = env.sample_band(rng, (1.4, 1.8), 1000)
    a, e, _ = env.elements(qp)
    assert a.min() >= 1.4 - 1e-9 and a.max() <= 1.8 + 1e-9
    assert e.max() <= 0.3 + 1e-9 and e.max() > 0.25
    E = env.energy(qp)
    np.testing.assert_allclose(E, -1 / (2 * a), rtol=1e-12)


def test_circular_orbit_has_expected_period(env):
    qp = env.from_elements(1.0, 0.0, 0.0)[None]
    steps = int(round(2 * np.pi / env.dt))  # period = 2*pi for a = 1
    for _ in range(steps):
        qp = env.step(qp)
    np.testing.assert_allclose(qp[0, :2], [1.0, 0.0], atol=0.03)  # step grid doesn't land exactly on 2*pi


def test_energy_and_angular_momentum_conserved(env):
    qp0 = env.sample_band(np.random.default_rng(2), (0.8, 1.8), 8)
    drift, _ = relative_drift(env, qp0, 10_000)
    assert drift.max() < 1e-6, drift
    _, _, L0 = env.elements(qp0)
    qp = qp0
    for _ in range(500):
        qp = env.step(qp)
    np.testing.assert_allclose(env.elements(qp)[2], L0, atol=1e-10)


def test_thrust_raises_energy_when_prograde(env):
    qp = env.from_elements(1.0, 0.0, 0.0)[None]  # at (1, 0) moving +y
    u = np.array([[0.0, env.u_max]])
    after = env.step(qp, u)
    assert env.energy(after)[0] > env.energy(qp)[0]
