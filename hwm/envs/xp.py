"""Tiny array-namespace shim so env formulas are written once and run on numpy or torch.

Only functions whose positional signatures agree between numpy and torch are used in env formulas:
``xp.sin/cos/stack(seq, dim)/zeros_like/ones_like`` plus operators and ``.sum(-1)``.
"""

import numpy as np
import torch


def xp_of(x):
    return torch if torch.is_tensor(x) else np


def solve_small(M, b):
    """Solve M x = b for batches of 1x1 or 2x2 matrices in closed form (numpy or torch).

    Batched ``np.linalg.solve`` on tiny matrices is dominated by call overhead.
    """
    n = M.shape[-1]
    xp = xp_of(M)
    if n == 1:
        return b / M[:, 0, :]
    if n == 2:
        a, c, d, e = M[:, 0, 0], M[:, 0, 1], M[:, 1, 0], M[:, 1, 1]
        det = a * e - c * d
        x0 = (e * b[:, 0] - c * b[:, 1]) / det
        x1 = (a * b[:, 1] - d * b[:, 0]) / det
        return xp.stack([x0, x1], -1)
    if xp is np:
        return np.linalg.solve(M, b[..., None])[..., 0]
    return torch.linalg.solve(M, b)


def mat2(a, b, c, d):
    """Batched 2x2 matrix [[a, b], [c, d]] from (B,) arrays."""
    if not torch.is_tensor(a):
        out = np.empty(np.shape(a) + (2, 2))
        out[..., 0, 0], out[..., 0, 1], out[..., 1, 0], out[..., 1, 1] = a, b, c, d
        return out
    return torch.stack([torch.stack([a, b], -1), torch.stack([c, d], -1)], -2)
