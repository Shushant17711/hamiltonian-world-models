"""Model E: Hamiltonian world model (ours; design §5, Req 5.1-5.3).

The latent is canonical z = (q, p) in R^{2n} (n = the env's degrees of freedom; ablation n_lat = 2n).
The learned Hamiltonian is

    H(q, p) = 1/2 p^T A(q) p + V(q),     A(q) = L(q) L(q)^T + eps I

with L lower-triangular from an MLP (softplus diagonal), so A is symmetric positive definite by
construction; the separable variant learns a constant A. Coordinates listed in ``angle_dims`` are
seen through (cos, sin) by every q-network.

Control and dissipation are port-Hamiltonian: an input matrix G(q) (n x d_u) and a damping matrix
R(q) = K(q) K(q)^T >= 0. One step of length h is a Strang splitting:

    p <- p + h/2 (G u - R A p)  ->  symplectic flow of H for h  ->  p <- p + h/2 (G u - R A p)

The conservative flow is leapfrog when separable, else the implicit midpoint rule with 6 unrolled,
differentiable fixed-point iterations. With u = 0 and R = 0 the map is symplectic (up to the
fixed-point tolerance), which is what bounds the learned energy over long rollouts (Req 5.3).

Loss (design §5): the shared open-loop rollout loss (sum_k ||dec(z_k) - o_k||^2 + reconstruction of
the context) + lambda_ae ||dec(enc(o_k)) - o_k||^2 + lambda_lat ||z_k - sg(enc(o_k))||^2 over the
window's targets. State mode only here; pixel mode arrives with task 10.1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hwm.integrators.torch_integrators import implicit_midpoint, leapfrog
from hwm.models.base import WorldModel
from hwm.models.nets import StateDecoder, StateEncoder, mlp
from hwm.models.registry import register


def tril_from_vector(v: Tensor, n: int) -> Tensor:
    """(..., n(n+1)/2) -> (..., n, n) lower-triangular with a softplus (positive) diagonal."""
    rows, cols = torch.tril_indices(n, n, device=v.device)
    L = v.new_zeros(*v.shape[:-1], n, n)
    L[..., rows, cols] = v
    diag = torch.arange(n, device=v.device)
    L[..., diag, diag] = F.softplus(L[..., diag, diag]) + 1e-4
    return L


def mlp_forward(net: nn.Sequential, x: Tensor) -> tuple[Tensor, list[Tensor]]:
    """Forward pass of a ``nets.mlp`` (Linear/SiLU stack, no LayerNorm) keeping pre-activations."""
    mods = list(net)
    pre, h = [], x
    for lin in mods[0:-1:2]:
        a = lin(h)
        pre.append(a)
        h = F.silu(a)
    return mods[-1](h), pre


def mlp_vjp(net: nn.Sequential, pre: list[Tensor], cot: Tensor) -> Tensor:
    """Input gradient cot^T d net(x) / dx from the cached pre-activations, built from forward ops only.

    Writing the backward pass by hand keeps dH/dq differentiable by a *single* reverse pass when
    training (no create_graph double backward), which is what makes model E affordable.
    """
    mods = list(net)
    g = cot @ mods[-1].weight
    for lin, a in zip(reversed(mods[0:-1:2]), reversed(pre), strict=True):
        s = torch.sigmoid(a)
        g = (g * (s * (1 + a * (1 - s)))) @ lin.weight  # silu'(a) = s (1 + a (1 - s))
    return g


@register("hamiltonian")
class HamiltonianModel(WorldModel):
    def __init__(self, cfg, env, obs_mode: str = "state"):
        if obs_mode != "state":
            raise NotImplementedError("model E pixel mode arrives with task 10.1")
        n = int(cfg.get("n_lat") or env.n)
        super().__init__(env.d_obs, env.d_u, d_z=2 * n, obs_mode=obs_mode)
        self.n, self.dt = n, env.dt
        hidden, layers = cfg.get("hidden", 256), cfg.get("layers", 3)
        h_hidden, h_layers = cfg.get("h_hidden", 128), cfg.get("h_layers", 2)
        dims = cfg.get("angle_dims")
        self.angle_dims = tuple(env.angle_dims if dims is None and n == env.n else (dims or ()))
        if any(i >= n for i in self.angle_dims):
            raise ValueError(f"angle_dims {self.angle_dims} out of range for n_lat = {n}")
        self.separable = bool(cfg.get("separable", False))
        self.eps = float(cfg.get("eps", 1e-3))
        self.mp_iters = int(cfg.get("midpoint_iters", 6))
        self.lambda_ae = float(cfg.get("lambda_ae", 1.0))
        self.lambda_lat = float(cfg.get("lambda_lat", 0.1))
        self.dissipation = bool(cfg.get("dissipation", True))
        # torch.compile the step on CUDA: one step is ~600 tiny kernels (7 field evaluations), and
        # fusing them is a ~5x speed-up. CPU (the test suite) stays eager.
        self.compile = bool(cfg.get("compile", True))
        self._compiled = None

        d_phi = n + len(self.angle_dims)
        n_tri = n * (n + 1) // 2
        self.V = mlp(d_phi, 1, h_hidden, h_layers)
        if self.separable:
            self.L_const = nn.Parameter(torch.zeros(n_tri))
        else:
            self.L_net = mlp(d_phi, n_tri, h_hidden, h_layers)
        self.G_net = mlp(d_phi, n * self.d_u, h_hidden, h_layers)
        self.K_net = mlp(d_phi, n * n, h_hidden, h_layers)
        nn.init.zeros_(self.K_net[-1].weight)  # start without dissipation; it must be learned
        nn.init.constant_(self.K_net[-1].bias, 0.0)
        self.enc = StateEncoder(env.d_obs, 2 * n, 1, hidden, layers)
        self.dec = StateDecoder(2 * n, env.d_obs, hidden, layers)

    # --- Hamiltonian pieces (q: (B, n)) --------------------------------------------------------------
    def features(self, q: Tensor) -> Tensor:
        if not self.angle_dims:
            return q
        a = list(self.angle_dims)
        rest = [i for i in range(self.n) if i not in self.angle_dims]
        return torch.cat([torch.cos(q[:, a]), torch.sin(q[:, a]), q[:, rest]], dim=-1)

    def features_vjp(self, q: Tensor, g: Tensor) -> Tensor:
        """g^T d features / dq for a cotangent g on the features."""
        if not self.angle_dims:
            return g
        a = list(self.angle_dims)
        rest = [i for i in range(self.n) if i not in self.angle_dims]
        k = len(a)
        out = torch.zeros_like(q)
        out[:, a] = -torch.sin(q[:, a]) * g[:, :k] + torch.cos(q[:, a]) * g[:, k : 2 * k]
        out[:, rest] = g[:, 2 * k :]
        return out

    def dH(self, q: Tensor, p: Tensor) -> tuple[Tensor, Tensor]:
        """(dH/dq, dH/dp) in closed form through the networks (forward ops only).

        With w = L^T p:  H = 1/2 |w|^2 + eps/2 |p|^2 + V,  dH/dp = L w + eps p,
        dH/dL_ij = p_i w_j on the lower triangle (times softplus' on the diagonal), pulled back
        through the L network and the angle features.
        """
        n = self.n
        phi = self.features(q)
        v_out, v_pre = mlp_forward(self.V, phi)
        g_phi = mlp_vjp(self.V, v_pre, torch.ones_like(v_out))
        if self.separable:
            L = tril_from_vector(self.L_const, n).expand(q.shape[0], n, n)
        else:
            raw, l_pre = mlp_forward(self.L_net, phi)
            L = tril_from_vector(raw, n)
        w = (L.transpose(-1, -2) @ p[:, :, None]).squeeze(-1)
        dHdp = (L @ w[:, :, None]).squeeze(-1) + self.eps * p
        if not self.separable:
            rows, cols = torch.tril_indices(n, n, device=q.device)
            cot = p[:, rows] * w[:, cols]
            diag = rows == cols
            cot = torch.where(diag, cot * torch.sigmoid(raw), cot)
            g_phi = g_phi + mlp_vjp(self.L_net, l_pre, cot)
        return self.features_vjp(q, g_phi), dHdp

    def field(self, z: Tensor) -> Tensor:
        """Hamiltonian vector field (dH/dp, -dH/dq) of the latent state."""
        dHdq, dHdp = self.dH(z[:, : self.n], z[:, self.n :])
        return torch.cat([dHdp, -dHdq], dim=-1)

    def A(self, q: Tensor) -> Tensor:
        """Inverse mass matrix (B, n, n), symmetric positive definite."""
        if self.separable:
            L = tril_from_vector(self.L_const, self.n).expand(q.shape[0], self.n, self.n)
        else:
            L = tril_from_vector(self.L_net(self.features(q)), self.n)
        eye = torch.eye(self.n, dtype=q.dtype, device=q.device)
        return L @ L.transpose(-1, -2) + self.eps * eye

    def potential(self, q: Tensor) -> Tensor:
        return self.V(self.features(q)).squeeze(-1)

    def H(self, q: Tensor, p: Tensor) -> Tensor:
        return 0.5 * (p[:, None, :] @ self.A(q) @ p[:, :, None]).reshape(-1) + self.potential(q)

    def G(self, q: Tensor) -> Tensor:
        return self.G_net(self.features(q)).reshape(-1, self.n, self.d_u)

    def R(self, q: Tensor) -> Tensor:
        K = self.K_net(self.features(q)).reshape(-1, self.n, self.n)
        return K @ K.transpose(-1, -2)

    # --- dynamics ------------------------------------------------------------------------------------
    def conservative_step(self, z: Tensor, h: float) -> Tensor:
        """Symplectic flow of H for one step of length h."""
        n = self.n
        if self.separable:
            A = self.A(z[:, :n])  # constant in q

            def dVdq(q: Tensor) -> Tensor:
                phi = self.features(q)
                out, pre = mlp_forward(self.V, phi)
                return self.features_vjp(q, mlp_vjp(self.V, pre, torch.ones_like(out)))

            q, p = leapfrog(dVdq, lambda p: (A @ p[:, :, None]).squeeze(-1), z[:, :n], z[:, n:], h)
            return torch.cat([q, p], dim=-1)
        return implicit_midpoint(self.field, z, h, iters=self.mp_iters)

    def _port_half(self, z: Tensor, u: Tensor, h: float) -> Tensor:
        q, p = z[:, : self.n], z[:, self.n :]
        dp = (self.G(q) @ u[:, :, None]).squeeze(-1)
        if self.dissipation:
            dp = dp - (self.R(q) @ self.A(q) @ p[:, :, None]).squeeze(-1)
        return torch.cat([q, p + 0.5 * h * dp], dim=-1)

    # --- WorldModel contract -------------------------------------------------------------------------
    def encode(self, ctx: Tensor) -> Tensor:
        return self.enc(ctx[:, -1:])

    def step(self, z: Tensor, u: Tensor) -> Tensor:
        if self.compile and z.is_cuda:
            if self._compiled is None:
                self._compiled = torch.compile(self._step)
            return self._compiled(z, u)
        return self._step(z, u)

    def _step(self, z: Tensor, u: Tensor) -> Tensor:
        z = self._port_half(z, u, self.dt)
        z = self.conservative_step(z, self.dt)
        return self._port_half(z, u, self.dt)

    def decode(self, z: Tensor) -> Tensor:
        return self.dec(z)

    def energy(self, z: Tensor) -> Tensor:
        return self.H(z[:, : self.n], z[:, self.n :])

    def extra_loss(self, batch, ro, horizon: int) -> dict[str, Tensor]:
        tgt = batch.target[:, :horizon]
        B, H = tgt.shape[:2]
        z_t = self.enc(tgt.reshape(B * H, 1, -1)).reshape(B, H, -1)
        ae = self.rollout_loss(self.dec(z_t), tgt)
        lat = self.rollout_loss(ro.z[:, 1:], z_t.detach())
        return {"ae": self.lambda_ae * ae, "lat": self.lambda_lat * lat}
