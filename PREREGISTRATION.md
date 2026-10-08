# Pre-registration — Hamiltonian World Models

Frozen on 2026-10-08, before any model, trainer or training run exists in this repository (the
commit that adds this file contains no model code; check `git log -- PREREGISTRATION.md hwm/models`).
It fixes the hypotheses, the metrics, the protocol and the decision rules for H1–H4. The verdicts in
`results/verdicts.md` are produced mechanically by `hwm/eval/hypotheses.py` from these rules; the same
numbers live as constants in `hwm/eval/thresholds.py`, and `tests/test_preregistration.py` fails if
the two copies disagree or if anything above the *Amendments* heading changes.

Source: design §7 (metrics) and §9 (thresholds) in `.kiro/specs/hamiltonian-world-models/design.md`.
Where §7/§9 left a choice open, this document closes it; those choices are marked **[fixed here]**.

## 1. Question

Does building physics into a world model's **structure** (a learned Hamiltonian, port-Hamiltonian
control and damping, and a symplectic integrator: model E) beat learning it through a **loss**
(model C, a latent Neural ODE with an energy penalty) and generic models (A MLP, B PINN, D RSSM) on
long-horizon energy behaviour, generalisation to unseen energies, and sample-efficient control? And
where does that advantage stop holding on a chaotic system?

## 2. Models compared

| id | model | obs modes | role |
|---|---|---|---|
| A | MLP residual next-state | state | baseline |
| B | PINN (A + residual against the true equations of motion) | state | baseline, **privileged** (uses the analytic dynamics) |
| C | latent Neural ODE, RK4, energy penalty | state | baseline (the loss-based approach) |
| D | RSSM (Dreamer-style) | state, pixels | baseline |
| E | Hamiltonian world model (ours) | state, pixels | tested model |
| E-ens | 5 independently initialised, bootstrapped E members | state, pixels | tested model (planning) |

**[fixed here]** The confirmatory E is the default configuration `configs/model/hamiltonian.yaml`
(latent dimension 2n, non-separable H with implicit midpoint). The separable variant and the
n_lat = 2n ablation are exploratory and cannot be substituted into a verdict.

**[fixed here]** Every model is trained with the shared trainer defaults of design §6 (AdamW, lr 3e-4,
cosine schedule, the 4→8→16→32 horizon curriculum, 30k steps state / 60k steps pixels, early stopping
on val 32-step nMSE) and evaluated from `ckpt_best.pt`. Any hyperparameter tuning uses the `val` split
only, gets the same number of trials for every model, and is logged in `LIMITATIONS.md`. The test
splits (`test_in`, `test_ood`, `test_long`) are read only by the final evaluation.

## 3. Environments and splits

pendulum, cart-pole, acrobot, orbit, with the energy bands of design §2 (fractions of `E_ref`;
orbit: semi-major axis a):

| env | train / val / test_in | test_ood |
|---|---|---|
| pendulum | [0.10, 0.60] | [0.70, 0.95] |
| cartpole | [0.10, 0.60] | [0.70, 0.95] |
| acrobot | [0.05, 0.40] | [0.50, 0.80] |
| orbit (a) | [0.8, 1.2] | [1.4, 1.8] |

Data configs are those committed in `configs/data/` at the time of this file; their hashes key the
datasets (`data/<env>/<hash>/`).

## 4. Metric definitions

**Feature map.** Observations are (q, q̇). For error computation each angle coordinate (pendulum θ;
cart-pole θ; acrobot θ₁, θ₂) is replaced by (cos, sin); the other coordinates are used as they are.
Call the result φ(o).

**Rollout error.** A model is given its context (state: the observation at t = k−1; pixels: k = 3
frames) and the true action sequence, and rolls out open-loop. **[fixed here]** nMSE(h) is the error
*at step h*, not averaged over steps 1..h:

    nMSE(h) = mean over trajectories and features of (φ(ô_h) − φ(o_h))² / Var_train(φ(o))

with Var_train computed per feature on the train split. Horizons: h ∈ {10, 100} on `test_in` and
`test_ood`; h = 1000 on `test_long` only.

**Energy drift.** From 50 passive initial states per band, roll the model out 10,000 steps with
u = 0, decode, map to (q, p) with the env's `obs_to_qp`, and evaluate the **true** H:

    drift = median over initial states i of  max_t |H(x_t) − H(x_0)| / E_ref

**[fixed here]** H1 uses the initial states of the train band (`test_in`); the OOD-band drift is
reported as secondary. The learned-energy drift of C and E is logged but not used in any verdict.

**Divergence. [fixed here]** A rollout that produces a non-finite value has drift = ∞ for that initial
state (the median absorbs it unless more than half diverge). For nMSE, each trajectory's error is capped
at 100 and non-finite errors count as 100. Every table reports the number of diverged rollouts per model.
Drift values are floored at 1e-9 before any ratio is taken, so that a ratio is never ∞/0.

**Seeds and aggregation.** Each (env, model, obs mode) cell is trained with seeds {0, 1, 2}
**[fixed here: exactly these three]**; extra seeds may be reported separately as exploratory. Seeds
are paired across models by index (seed s of every model sees the same data and evaluation set).
Confidence intervals are percentile bootstraps over seeds, 10,000 resamples, α = 0.05, bootstrap RNG
seed 0 (`hwm.eval.stats.bootstrap_ci`). Ratios are bootstrapped in log space over the paired per-seed
log ratios, and differences over the paired per-seed differences.

*Consequence, stated in advance:* with 3 seeds, the 2.5th percentile of a resampled mean equals the
smallest per-seed value (all three draws hitting it has probability 1/27 > 2.5%). So "CI lower bound
> x" here means "every seed exceeds x". That is the intended, conservative reading.

## 5. Hypotheses and decision rules

### H1 — energy drift (state mode)

On each of pendulum, cart-pole and orbit (acrobot is reserved for H4), let r_s = drift_C,s / drift_E,s
for seed s. The env **passes** if the geometric mean of r_s over seeds is ≥ 10 **and** the lower bound
of its 95% bootstrap CI is > 3. **H1 is supported if ≥ 2 of the 3 envs pass**; otherwise not supported.

### H2 — out-of-distribution accuracy (state mode)

On `test_ood` at h = 100, for each baseline X ∈ {A, B, C, D(state)} and seed s, d_X,s = nMSE_X,s −
nMSE_E,s. On an env, E **beats** X if the 95% bootstrap CI of mean_s d_X,s lies entirely above 0. The
env **passes** if E beats all four baselines. **H2 is supported if ≥ 3 of the 4 envs pass.**
B's privileged status is printed next to every B comparison.

### H3 — sample efficiency of control (pixel mode)

On pendulum, cart-pole and orbit in pixel mode, run the model-based RL loop of design §8 (5 random
episodes, then alternate 2k fine-tuning steps and 1 MPC episode; CEM horizon 30, population 400,
elites 40, 5 iterations). E-ens plans with disagreement penalty β = 1.0; the RSSM plans as a single
model with β = 0. At cumulative env steps {1k, 2k, 5k, 10k, 20k, 50k} (random episodes included),
evaluate 10 MPC episodes with the success predicates of design §2.

**[fixed here]** For each seed, N80 = the first checkpoint whose success rate is ≥ 0.8 (8 of 10), or
∞ if none is reached within 50k. Per model, N80 is the median over seeds. The task **passes** if
N80(E-ens) ≤ 0.5 × N80(RSSM), where finite ≤ 0.5 × ∞ is true and ∞ ≤ 0.5 × ∞ is false (if neither
model learns the task, that is not evidence for E). **H3 is supported if ≥ 2 of the 3 tasks pass.**

### H4 — the chaotic acrobot (descriptive, with a stated prediction)

Estimate the acrobot's maximal Lyapunov exponent λ from the GL4 simulator (Benettin two-trajectory
renormalisation, d₀ = 1e-8, renormalised every step, 20 initial states per band); the Lyapunov time is
t_λ = 1/λ, per band.

**[fixed here]** advantage(h) = nMSE_baseline(h) / nMSE_E(h), where the baseline is the strongest one
at that h (lowest seed-mean nMSE among A, B, C, D, state mode); per-baseline curves are also reported.
Horizon grid h ∈ {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000} steps on the train-band `test_long`
trajectories, plotted against h·dt / t_λ; h ≤ 100 is repeated on `test_in` (actuated) as a secondary
curve. CIs come from the paired log-ratio bootstrap over seeds.

Prediction: (a) the CI of advantage(h) lies above 1 at every grid h with h·dt < 1 t_λ, and (b) the
CI includes 1 at every grid h with h·dt > 3 t_λ. The verdict is *confirmed* if both (a) and (b) hold,
*partially confirmed* if exactly one does, and *not confirmed* otherwise. If no grid h falls in a
regime, that part is reported as *untestable*. The H1–H3 metrics are also computed on the acrobot and
reported without thresholds.

## 6. What is not a deviation, and what is

- Adding exploratory analyses, ablations, figures or seeds is allowed, as long as they are labelled
  exploratory and the confirmatory verdicts above are reported unchanged.
- Fixing a bug in evaluation code is allowed; if it changes any number, the old and new numbers are
  both listed under *Amendments*.
- Changing a threshold, a metric definition, a split, a seed set or a decision rule after any result
  exists is a deviation. It is recorded under *Amendments* with its date and reason, and the verdict
  under the original rule is still reported first.

## 7. Machine-readable copy

`tests/test_preregistration.py` checks that this block equals `hwm.eval.thresholds.THRESHOLDS`.

```yaml
seeds: [0, 1, 2]
bootstrap:
  n_resamples: 10000
  alpha: 0.05
  rng_seed: 0
metrics:
  nmse_cap: 100.0
  drift_floor: 1.0e-9
  drift_steps: 10000
  drift_initial_states: 50
  rollout_horizons: [10, 100, 1000]
H1:
  envs: [pendulum, cartpole, orbit]
  obs_mode: state
  band: test_in
  min_ratio: 10.0
  min_ci_lower: 3.0
  min_envs_passing: 2
H2:
  envs: [pendulum, cartpole, acrobot, orbit]
  obs_mode: state
  split: test_ood
  horizon: 100
  baselines: [A, B, C, D]
  min_envs_passing: 3
H3:
  envs: [pendulum, cartpole, orbit]
  obs_mode: pixels
  success_threshold: 0.8
  eval_episodes: 10
  checkpoints: [1000, 2000, 5000, 10000, 20000, 50000]
  max_ratio: 0.5
  beta_ensemble: 1.0
  min_envs_passing: 2
H4:
  env: acrobot
  horizons: [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
  early_max_lyapunov_times: 1.0
  late_min_lyapunov_times: 3.0
  lyapunov_d0: 1.0e-8
  lyapunov_initial_states: 20
```

## Amendments

None.
