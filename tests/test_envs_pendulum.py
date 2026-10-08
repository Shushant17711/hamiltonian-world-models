import numpy as np
import pytest

from hwm.envs import make


@pytest.fixture
def env():
    return make("pendulum")


def test_torque_changes_energy_and_is_clipped(env):
    qp = np.zeros((1, 2))
    big = env.step(qp, np.array([[100.0]]))
    capped = env.step(qp, np.array([[env.u_max]]))
    np.testing.assert_allclose(big, capped)
    assert env.energy(capped)[0] > 0
