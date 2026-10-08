# Design — Hamiltonian World Models (hwm)

Source: `PLAN.md` (rough plan) + `requirements.md`. Decisions here are final for v1; change them by
editing this file, not by drifting in code.

## §1 Overview

```
            ┌──────────── ground-truth simulators (numpy, GL4 symplectic) ────────────┐
            │  pendulum · cart-pole · acrobot · orbit   H(q,p) analytic, reward, success│
            └───────────────┬──────────────────────────────────────┬──────────────────┘
                            │ trajectories (.npz, state only)      │ online episodes
                            ▼                                      ▼
   torch SDF renderer ──► Dataset (windows, normalisation) ──► Trainer ──► results/<run>/
                                                                   │
              models A MLP · B PINN · C latent-ODE · D RSSM · E Hamiltonian · E-ens
                                                                   │
                         eval (rollout / energy / OOD / calibration / Lyapunov)
                         planning (CEM-MPC, model-based RL loop)
                         hypotheses.py ──► verdict tables ──► figures, README
```

## §2 Environments

All environments use SI-like units with m = l = g-normalised constants given below, dt = 0.05 s,
zero-order-hold actions, and actions clipped to `u_max`. State is canonical (q, p) internally;
the observation is (q, q̇) (Req 3.1). Angles are measured from the downward vertical; angle
observations are wrapped to (−π, π] only for rendering and reward, never inside the integrator.

| env | q | H(q, p) | input B·u | u_max | E_ref |
|---|---|---|---|---|---|
| pendulum | θ | p²/(2ml²) + mgl(1−cos θ) | ṗ += u | 2.0 | 2mgl |
| cart-pole | (x, θ) | ½pᵀM(q)⁻¹p + m_p g l(1−cos θ), M = [[m_c+m_p, m_p l cos θ],[m_p l cos θ, m_p l²]] | ṗ_x += u | 10.0 | 2 m_p g l |
| acrobot | (θ₁, θ₂) (θ₂ relative) | ½pᵀM(q)⁻¹p + V(q), point masses, standard M₁₁ = (m₁+m₂)l₁² + m₂l₂² + 2m₂l₁l₂cos θ₂, M₁₂ = m₂l₂² + m₂l₁l₂cos θ₂, M₂₂ = m₂l₂², V = (m₁+m₂)gl₁(1−cos θ₁) + m₂gl₂(1−cos(θ₁+θ₂)) | ṗ₂ += u | 4.0 | V(π, 0) |
| orbit | (x, y) relative | ½‖p‖² − k/‖q‖ (μ = k = 1) | ṗ += u ∈ ℝ² | 0.05 | 1/(2·1) (|E| at a = 1) |

Constants: pendulum m = l = 1, g = 9.81. Cart-pole m_c = 1, m_p = 0.1, l = 0.5, g = 9.81. Acrobot
m₁ = m₂ = 1, l₁ = l₂ = 1, g = 9.81. Optional linear damping `−D q̇` added to ṗ, default D = 0.

**Ground-truth integrator (Req 1.3).** Each env implements `dH(q, p) -> (∂H/∂q, ∂H/∂p)` analytically
(checked against finite differences in tests). The full vector field
`q̇ = ∂H/∂p, ṗ = −∂H/∂q + B u − D ∂H/∂p` is integrated with 2-stage Gauss–Legendre (order 4,
symplectic for u = D = 0), with per-env substeps chosen so the 10k-step drift is ≥ 100× below the
Req 1.3 threshold (pendulum: 4; error scales as h⁴), solving the stage equations by fixed-point
iteration to tol 1e-13 (max 50 iterations). Integration is vectorised over a batch of trajectories
in numpy float64. One integrator for all four systems keeps the guarantee uniform.

**Energy-band sampling (Req 1.2, 2.2).** `reset(rng, band=(lo, hi))` draws a target energy
E* ~ U(lo, hi)·E_ref, then draws a random configuration with V(q) < E* (rejection) and a random
momentum direction, scaled so that H(q, p) = E* exactly (T is quadratic in p). Orbit: draw the
semi-major axis a from the band, eccentricity e ~ U(0, 0.3), random phase and orientation, and set
(q, p) from the Kepler elements. Bands:

| env | train / test_in | test_ood |
|---|---|---|
| pendulum | [0.10, 0.60] | [0.70, 0.95] |
| cart-pole | [0.10, 0.60] | [0.70, 0.95] |
| acrobot | [0.05, 0.40] | [0.50, 0.80] |
| orbit (a) | [0.8, 1.2] | [1.4, 1.8] |

Cart-pole: initial states have zero horizontal momentum p_x (conserved when u = 0, so passive
rollouts stay on the track; the cart's kinetic-energy share is then far below the 20% cap), and trajectories whose
|x| exceeds 3.0 are rejected and resampled (the fixed camera covers |x| ≤ 3.2).

**Tasks (Req 1.4).** Each env defines `reward(obs, u)` (dense, shaped) and `success(episode_obs)`.

| env | start | episode | success |
|---|---|---|---|
| pendulum | hanging, θ̇ = 0 | 200 steps | cos θ < −0.95 on every one of the last 50 steps |
| cart-pole | hanging | 250 steps | cos θ < −0.95 and |x| < 2 on the last 50 steps |
| acrobot | hanging | 300 steps | tip height ≥ 1.5 (l₁ + l₂ = 2) at any step |
| orbit | circular r = 1 | 400 steps | |r − 1.5| < 0.05 and |ṙ| < 0.05 on the last 50 steps |

## §3 Data

`scripts/gen_data.py --config configs/data/<env>.yaml` writes
`data/<env>/<hash>/{train,val,test_in,test_ood,test_long}.npz` (Req 2.3: hash = sha1 of the
resolved data config). Arrays: `obs (N, T+1, d_obs)`, `qp (N, T+1, 2n)`, `act (N, T, d_u)`,
`energy (N, T+1)`, `passive (N,)` (bool).

- Actions: Ornstein–Uhlenbeck, θ = 0.15, σ = 0.3·u_max, clipped. 25% of trajectories are
  **passive** (u = 0), which gives Model C its energy-penalty data and gives every model the same data.
  Cart-pole adds a PD centring term (kp = kd = 3) on the cart position to the OU force, since pure
  OU forces drive ~90% of 200-step rollouts off the track; the applied, clipped actions are stored.
- Sizes: train 1000 × 200 steps, val 100 × 200, test_in / test_ood 200 × 200, test_long
  50 passive × 1000 steps per band. The 10k-step drift test needs only initial states.
- Images are **not stored**. `hwm.envs.render` is a vectorised torch SDF rasteriser (anti-aliased
  by a 1-px smoothstep), 64×64 grayscale in [0, 1], that runs on CPU or GPU from observations at
  batch time (Req 3.2). Pixel contexts stack k = 3 frames (Req 3.3).
  World windows (half-width): pendulum 1.3, cart-pole 3.2, acrobot 2.2, orbit 2.5. The cart-pole
  pole is only ~5 px long at this scale (the camera must cover the track); recorded in LIMITATIONS.
- `hwm.data.WindowDataset` yields random windows `(ctx_obs (k, ·), actions (H,), target_obs (H, ·),
  rewards (H,))` with H set by the curriculum. Normalisation statistics come from train only and
  are saved with the model.

## §4 Shared contracts

```python
# hwm/models/base.py
class WorldModel(nn.Module):
    obs_mode: Literal["state", "pixels"]
    context: int                                  # 1 for state, 3 for pixels
    def encode(self, ctx: Tensor) -> Tensor        # (B, k, *obs) -> z (B, d_z)
    def step(self, z: Tensor, u: Tensor) -> Tensor # one dt
    def decode(self, z: Tensor) -> Tensor          # -> obs (B, *obs)
    def reward(self, z: Tensor, u: Tensor) -> Tensor | None   # pixel mode: learned head
    def energy(self, z: Tensor) -> Tensor | None              # learned energy, if any
    def rollout(self, ctx, actions) -> Rollout     # default: encode, then step/decode loop
    def loss(self, batch: Batch, horizon: int) -> tuple[Tensor, dict[str, float]]
```

`Rollout = (z: (B, H+1, d_z), obs: (B, H+1, *obs), reward: (B, H) | None)`.
`Ensemble(WorldModel)` holds M members; `rollout` returns stacked `(M, B, …)` plus `mean` and
`disagreement` (per-step std over members of the decoded obs, averaged over dims).

Config: YAML → nested frozen dataclasses (`hwm.config`), CLI overrides `a.b=c`. Model registry
`hwm.models.build(cfg) -> WorldModel`, env registry `hwm.envs.make(name)`.

## §5 Models

Shared building blocks (`hwm.models.nets`): MLP (SiLU, LayerNorm optional), state encoder/decoder MLPs,
CNN encoder (4 × conv stride 2, 32-64-128-256 → linear) and transposed-CNN decoder for 64×64.
Default widths: hidden 256, 3 layers. Every model is < 5M params (asserted in the trainer, Req 6.3).

- **A — MLP.** `z = normalised obs`; `step: z + f([z, u])`. In pixel mode A is not run (it has no
  latent); A, B and C are state-only, as in PLAN phase 2.
- **B — PINN.** A plus `λ_phys · ‖(v̂_{t+1} − v̂_t)/dt − a_true(q̂_mid, v̂_mid, u)‖²` on predicted
  states, and the same residual on 256 random collocation states per batch drawn from the train
  state box. `a_true` uses the env's analytic dynamics, which is privileged knowledge and is
  labelled so in every table.
- **C — latent Neural ODE + energy penalty** (re-implementation of the friend's model, plus actions).
  Encoder MLP → z ∈ ℝ^{d_z} (d_z = 2n by default, ablation 4n); `ż = f_θ(z, u)`, RK4 with 1
  substep; energy head E_ψ(z). Loss = reconstruction + multi-step prediction +
  `λ_E · Var_t E_ψ(z_t)` on passive trajectories only, + `λ_E · (E_ψ(enc(x)) − H_true(x))²`
  supervision, as in the original. The energy penalty is a loss, not a structural constraint.
- **D — RSSM.** GRU h = 200, stochastic s = 30 (diagonal Gaussian), prior/posterior MLPs, KL
  balancing α = 0.8, free nats 1.0. Open-loop rollouts use the prior. Decoder from [h, s]. Runs in
  state and pixel modes.
- **E — Hamiltonian world model (ours).**
  - Latent z = (q, p) ∈ ℝ^{2n}, where n = the env's degrees of freedom (a structural prior; ablation
    with n_lat = 2n). Config `angle_dims` marks the latent coordinates that are angles; V_θ and
    M_θ⁻¹ see them through (cos, sin) features.
  - H_θ(q, p) = ½ pᵀ A_θ(q) p + V_θ(q), with A_θ(q) = L_θ(q)L_θ(q)ᵀ + εI (Cholesky factor from
    an MLP, softplus diagonal) — symmetric positive definite by construction. Separable variant:
    A_θ a learned constant.
  - Port-Hamiltonian input and dissipation: G_θ(q) ∈ ℝ^{n×d_u} (MLP), R_θ(q) = K Kᵀ ⪰ 0.
  - Step, Strang splitting (Req 5.2):
    `p ← p + (h/2)(G u − R A p)` → symplectic flow of H_θ for h → `p ← p + (h/2)(G u − R A p)`.
    Conservative flow: leapfrog if separable, else implicit midpoint with 6 unrolled fixed-point
    iterations (differentiable), initialised by one explicit Euler step. With u = 0 and R = 0 the
    map is exactly symplectic up to fixed-point tolerance (Req 5.3).
  - State-mode encoder: MLP(obs) → (q, p). Decoder: MLP(q, p) → obs.
  - Pixel mode: CNN(3 frames) → (q, p); **the image decoder sees q only**, since images depend only
    on configuration. A reward head r(q, p, u) is used for planning in pixel mode.
  - Loss: Σ_k ‖dec(z_k) − o_k‖² + λ_ae‖dec(enc(o_k)) − o_k‖² + λ_lat‖z_k − sg(enc(ctx_k))‖² (+ reward MSE).
- **E-ens.** 5 members, each with its own init seed and its own bootstrap resample of the train
  trajectories (Req 5.4).

## §6 Training

One `Trainer` for every model (Req 6.1): AdamW, lr 3e-4, cosine schedule, grad-clip 10, batch 128
windows, bf16 autocast on CUDA for CNNs only (integrators run in fp32). Horizon curriculum
H: 4 → 8 → 16 → 32 at 0 / 20 / 40 / 60% of steps. Default 30k steps for state models and 60k for
pixel models; early stopping on val 32-step nMSE. Writes `results/<run_id>/{config.yaml,
metrics.jsonl, ckpt.pt, ckpt_best.pt}`, with `run_id = <env>-<model>-<obs>-s<seed>`, and resumes
if `ckpt.pt` exists (Req 6.2). Seeds cover torch, numpy and python.

## §7 Evaluation

- **Rollout error (7.1):** nMSE(h) = mean over trajectories and dims of (ô − o)² / Var_train(o),
  for h ∈ {10, 100, 1000}, on test_in, test_ood and test_long (h = 1000 only on test_long).
  Angles are compared through (cos, sin) differences, so wrapping does not inflate the error.
- **Energy drift (7.2):** from 50 passive initial states per band, roll the model out 10,000 steps
  with u = 0, decode, convert to (q, p) with the env's `obs_to_qp`, and compute
  `drift = median_i max_t |H(x_t) − H(x_0)| / E_ref` using the **true** H. Also logged: the
  learned-energy drift for C and E.
- **Aggregation (7.4):** `hwm.eval.stats.bootstrap_ci(values, n=10_000, alpha=0.05)` over seeds
  (≥ 3), plus ratio CIs by paired bootstrap.
- **Calibration (8.4):** per (trajectory, horizon) pair, ensemble disagreement vs. true error;
  Spearman ρ and a 10-bin reliability curve.
- **Lyapunov (9.2):** Benettin two-trajectory renormalisation on the GL4 simulator, d₀ = 1e-8,
  renormalised every step, averaged over 20 initial states per band.
- **Hypotheses (7.5, 8.5, 9.1):** `hwm.eval.hypotheses` reads `results/**/eval/*.json` and applies
  the thresholds in PREREGISTRATION.md (§9) mechanically, writing `results/verdicts.md`.

## §8 Planning

- **CEM-MPC (8.1, 8.2):** horizon 30, population 400, elites 40, 5 iterations, momentum 0.1,
  warm-started by shifting the previous plan. Cost = −Σ r̂_t + β Σ disagreement_t (β = 1.0 by
  default; β = 0 for single models). State mode: r̂ = env.reward(decoded obs, u). Pixel mode: the
  learned reward head (ensemble: member mean). All candidates run as one batched rollout on the GPU.
- **Model-based RL loop (8.3):** 5 random-action episodes to start; then repeat {train / fine-tune
  2k steps on all data → collect 1 MPC episode}. Evaluate 10 MPC episodes when the cumulative env
  steps reach {1k, 2k, 5k, 10k, 20k, 50k}. Log `(env_steps, success_rate, mean_return)` to
  `results/<run>/mbrl.jsonl`.

## §9 Pre-registered thresholds (copied into PREREGISTRATION.md, Req 10.2)

- **H1:** on each of pendulum, cart-pole, orbit (state mode), the seed-geometric-mean ratio
  drift(C) / drift(E) ≥ 10, and the lower bound of its 95% bootstrap CI > 3. H1 holds if this is
  true on ≥ 2 of the 3 envs.
- **H2:** on test_ood at h = 100, E's nMSE is lower than each of A, B, C, D(state), with the 95% CI
  of the paired difference excluding 0, on ≥ 3 of 4 envs.
- **H3:** in pixel mode on pendulum, cart-pole and orbit, the env steps E-ens needs to reach ≥ 80%
  success are ≤ 0.5× what RSSM needs (∞ if never reached within 50k), on ≥ 2 of 3 tasks.
- **H4 (descriptive, with a stated prediction):** on the acrobot, advantage(h) = nMSE_baseline(h) /
  nMSE_E(h) is plotted against h / t_λ (Lyapunov time). Prediction: advantage > 1 for h < 1 t_λ,
  and its CI includes 1 for h > 3 t_λ.

## §10 Testing strategy

pytest, CPU-only, total < 2 min (Req 10.5). Physics tests: finite-difference checks of dH, GL4
energy conservation (Req 1.3 thresholds), determinism, band sampling. Model tests: shape and
interface contract for every model, param budget, symplecticity of E's step (Jacobian J satisfies
JᵀΩJ = Ω to 1e-4 with u = R = 0), and that E's learned energy is bounded over 10k steps for a
randomly initialised H_θ (Req 5.3). Pipeline tests: a tiny end-to-end train (50 steps) per model,
CEM finding the optimum of a known quadratic, and hypothesis evaluation on synthetic result files.
GPU-heavy experiments live in scripts, never in tests.

Long-horizon checks (the 10k-step drift of Req 1.3, about 45 s on CPU for all four envs) carry
`@pytest.mark.slow` and run with `uv run pytest -m slow`; the default suite applies the same drift
thresholds over 2k steps so it stays under the 2-minute budget of Req 10.5. Measured 10k-step drift:
pendulum 3e-9, orbit 7e-10, cart-pole 4e-8, acrobot 9e-7 (substeps 4/4/4/8).

## §11 Repository layout

```
hwm/  config.py
      envs/{base,pendulum,cartpole,acrobot,orbit,render,registry}.py
      integrators/{gl4.py (numpy), torch_integrators.py (leapfrog, implicit_midpoint, rk4)}
      data/{generate,dataset}.py
      models/{base,nets,mlp,pinn,latent_ode,rssm,hamiltonian,ensemble,registry}.py
      train/trainer.py
      planning/{cem,mpc,mbrl}.py
      eval/{rollout,energy,stats,calibration,lyapunov,hypotheses,figures}.py
scripts/{gen_data,train,evaluate,run_mbrl,sweep,lyapunov,make_figures}.py
configs/{data,model,sweeps}/*.yaml   tests/   results/ (git-ignored except verdicts + figures)
README.md  LICENSE  LIMITATIONS.md  PREREGISTRATION.md  pyproject.toml (uv)
```

## §12 Risks

- Implicit-midpoint unrolling cost inside CEM (400 × 5 members × 30 steps × 6 iterations):
  mitigated by the separable fast path and by batching every candidate in one call.
  If the MBRL loop exceeds ~2 h per seed, reduce the population to 200 and record it in LIMITATIONS.
- A latent (q, p) learned from pixels need not be canonical; the Strang/symplectic structure holds
  in the learned coordinates regardless, which is exactly what H1 tests through the decoded true energy.
- Acrobot chaos can make every model look equally bad beyond a few Lyapunov times. That is the point of H4.
