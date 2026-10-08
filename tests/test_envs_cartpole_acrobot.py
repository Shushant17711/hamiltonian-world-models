import numpy as np

from hwm.envs import make


def test_cartpole_band_respects_cart_share_and_track():
    env = make("cartpole")
    qp = env.sample_band(np.random.default_rng(0), (0.7, 0.95), 300)
    q, v = qp[:, :2], env.qp_to_obs(qp)[:, 2:]
    T = env.H(q, qp[:, 2:]) - env.V(q)
    T_cart = 0.5 * (env.m_c + env.m_p) * v[:, 0] ** 2
    assert (T_cart <= env.max_cart_share * T + 1e-12).all()
    assert (np.abs(q[:, 0]) <= 1.5).all()
    assert env.valid_state(qp).all()
    assert not env.valid_state(np.array([[3.5, 0.0, 0.0, 0.0]])).any()


def test_cartpole_is_non_separable_and_force_moves_cart():
    env = make("cartpole")
    assert not env.separable
    qp = np.zeros((1, 4))
    after = env.step(qp, np.array([[env.u_max]]))
    assert env.qp_to_obs(after)[0, 2] > 0  # cart velocity increases with positive force


def test_acrobot_torque_only_at_elbow_and_tip_height():
    env = make("acrobot")
    np.testing.assert_array_equal(env.B, [[0.0], [1.0]])
    assert env.tip_height(np.array([0.0, 0.0])) == -2.0
    np.testing.assert_allclose(env.tip_height(np.array([np.pi, 0.0])), 2.0)
    # upright, at rest: V equals E_ref
    np.testing.assert_allclose(env.V(np.array([[np.pi, 0.0]])), [env.E_ref])
    # elbow torque from rest changes the elbow velocity first
    after = env.step(np.zeros((1, 4)), np.array([[env.u_max]]))
    v = env.qp_to_obs(after)[0, 2:]
    assert v[1] > 0 and abs(v[1]) > abs(v[0])
