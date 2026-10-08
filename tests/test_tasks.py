"""Rewards, success predicates and task starts (Req 1.4)."""

import numpy as np
import pytest
import torch

from hwm.envs import make, names


def _rollout(env, qp0, steps, u=None):
    qps = [qp0]
    for _ in range(steps):
        qps.append(env.step(qps[-1], u))
    return env.qp_to_obs(np.stack(qps, 1))  # (B, T+1, d_obs)


@pytest.mark.parametrize("name", names())
def test_task_start_and_hanging_episode_fails(name):
    env = make(name)
    qp0 = env.task_start(3)
    assert qp0.shape == (3, 2 * env.n)
    obs = _rollout(env, qp0, env.episode_len)  # passive: nothing happens
    assert not env.success(obs).any()
    assert env.success(obs).shape == (3,)


def _upright_obs(env, T):
    """A hand-built episode resting at the goal."""
    obs = np.zeros((1, T + 1, env.d_obs))
    if env.name == "pendulum":
        obs[..., 0] = np.pi
    elif env.name == "cartpole":
        obs[..., 1] = -np.pi  # angles are not wrapped
        obs[..., 0] = 1.0
    elif env.name == "acrobot":
        obs[..., -1, 0] = np.pi  # upright only on the final step: "at any step"
    else:  # orbit: circular at r = 1.5 (velocity tangential)
        obs[..., 0] = 1.5
        obs[..., 3] = np.sqrt(1 / 1.5)
    return obs


@pytest.mark.parametrize("name", names())
def test_success_on_goal_episode(name):
    env = make(name)
    obs = _upright_obs(env, env.episode_len)
    assert env.success(obs).all()


def test_success_requires_whole_hold_window():
    env = make("pendulum")
    obs = _upright_obs(env, 200)
    obs[0, -10, 0] = 0.0  # one hanging frame inside the last 50
    assert not env.success(obs)[0]
    env = make("cartpole")
    obs = _upright_obs(env, 250)
    obs[0, -1, 0] = 2.5  # off-centre at the end
    assert not env.success(obs)[0]


def test_orbit_transfer_is_feasible_with_thrust_limit():
    """A simple element-feedback controller (raise apoapsis, coast, circularise) reaches the target orbit."""
    env = make("orbit")
    qp = env.task_start(1)
    obs = [env.qp_to_obs(qp)]

    def prograde(qp):
        v = qp[:, 2:]
        return env.u_max * v / np.linalg.norm(v, axis=-1, keepdims=True)

    phase, e_prev = "raise", np.inf
    for _ in range(env.episode_len):
        a, e, _ = env.elements(qp)
        _, r_dot = env.radial(env.qp_to_obs(qp))
        if phase == "raise" and a[0] * (1 + e[0]) >= 1.5:
            phase = "coast"
        # start circularising slightly before apoapsis so the finite burn straddles it
        if phase == "coast" and r_dot[0] < 0.05:
            phase = "circularise"
        if phase == "circularise" and e[0] > e_prev:  # eccentricity bottomed out
            phase = "done"
        e_prev = e[0] if phase == "circularise" else np.inf
        u = prograde(qp) if phase in ("raise", "circularise") else None
        qp = env.step(qp, u)
        obs.append(env.qp_to_obs(qp))
    obs = np.stack(obs, 1)
    assert obs.shape[1] == env.episode_len + 1
    assert env.success(obs)[0], env.radial(obs[0, -1])


@pytest.mark.parametrize("name", names())
def test_reward_numpy_and_torch_agree_and_prefer_goal(name):
    env = make(name)
    rng = np.random.default_rng(0)
    obs = env.qp_to_obs(env.sample_band(rng, (0.1, 0.6) if name != "orbit" else (0.8, 1.2), 32))
    u = rng.uniform(-env.u_max, env.u_max, (32, env.d_u))
    r_np = env.reward(obs, u)
    r_t = env.reward(torch.as_tensor(obs), torch.as_tensor(u))
    assert r_np.shape == (32,)
    np.testing.assert_allclose(r_t.numpy(), r_np, rtol=1e-12)
    goal = _upright_obs(env, 0)[:, -1]
    start = env.qp_to_obs(env.task_start(1))
    zero = np.zeros((1, env.d_u))
    assert env.reward(goal, zero)[0] > env.reward(start, zero)[0]
