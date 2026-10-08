"""CEM planner (Req 8.1)."""

import pytest
import torch

from hwm.planning.cem import CEM, CEMConfig


def test_finds_quadratic_optimum():
    target = torch.linspace(-0.6, 0.8, 10)[:, None].repeat(1, 2)
    target[:, 1] *= -0.5
    cem = CEM(
        2,
        CEMConfig(horizon=10, population=500, elites=50, iterations=8),
        generator=torch.Generator().manual_seed(0),
    )
    seq, stats = cem.plan(lambda a: ((a - target) ** 2).sum((1, 2)))
    assert (seq - target).abs().max() < 0.08
    assert stats["best_cost"] < 0.03  # from ~3.2 for the zero sequence


def test_respects_bounds():
    cem = CEM(
        1,
        CEMConfig(horizon=5, population=100, elites=10, iterations=5),
        generator=torch.Generator().manual_seed(1),
    )
    seq, _ = cem.plan(lambda a: -(a**2).sum((1, 2)) - a.sum((1, 2)))  # pushes towards +inf
    assert seq.max() <= 1.0 and seq.min() >= -1.0 and seq.mean() > 0.9


def test_warm_start_shift_and_act():
    cem = CEM(1, CEMConfig(horizon=4, population=50, elites=5))
    cem.mean = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
    cem.shift()
    torch.testing.assert_close(cem.mean, torch.tensor([[0.2], [0.3], [0.4], [0.4]]))
    cem.reset()
    assert cem.mean.abs().sum() == 0
    seen = {}

    def cost(a):
        seen.setdefault("first", a[0].clone())
        return ((a - 0.5) ** 2).sum((1, 2))

    a, _ = cem.act(cost)
    assert a.shape == (1,) and abs(a.item() - 0.5) < 0.15
    torch.testing.assert_close(seen["first"], torch.zeros(4, 1))  # the warm start is evaluated first


def test_nonfinite_costs_are_never_elites():
    def cost(a):
        c = ((a - 0.3) ** 2).sum((1, 2))
        c[a[:, 0, 0] > 0.5] = float("nan")
        return c

    seq, _ = CEM(
        1, CEMConfig(horizon=3, population=200, elites=20), generator=torch.Generator().manual_seed(2)
    ).plan(cost)
    assert abs(seq[0].item() - 0.3) < 0.1


def test_bad_config():
    with pytest.raises(ValueError):
        CEM(1, CEMConfig(population=10, elites=20))


def test_batched_problems_are_independent():
    targets = torch.tensor([-0.5, 0.0, 0.7])[:, None, None]  # 3 problems with different optima
    cem = CEM(
        1,
        CEMConfig(horizon=4, population=300, elites=30, iterations=6),
        generator=torch.Generator().manual_seed(3),
        n=3,
    )

    def cost(a):  # a: (3 * P, T, 1), problem-major
        a = a.reshape(3, -1, 4, 1)
        return ((a - targets[:, None]) ** 2).sum((2, 3)).reshape(-1)

    seq, _ = cem.plan(cost)
    assert seq.shape == (3, 4, 1)
    torch.testing.assert_close(seq.mean((1, 2)), targets.reshape(3), atol=0.08, rtol=0)
    a, _ = cem.act(cost)
    assert a.shape == (3, 1)
