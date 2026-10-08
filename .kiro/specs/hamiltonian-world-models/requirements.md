# Requirements — Hamiltonian World Models (hwm)

## Introduction

An action-conditioned world model whose architecture guarantees energy conservation when there is
no control and no damping. It works from state vectors and from 64×64 images and is used for
model-predictive control. The research question: does building physics into the **structure**
(a learned Hamiltonian plus a symplectic integrator) beat learning it through a **loss** on
long-horizon accuracy, generalisation to unseen energies, and sample-efficient control?

Baseline being improved on: Optimus2007/physics-informed-world-models (one 2-body trajectory,
MLP / PINN / latent Neural ODE + energy penalty, no actions, no OOD test, no uncertainty, no control).

Hardware budget: RTX 4060 Laptop (8 GB VRAM). Every model has < 5M parameters.

---

### Requirement 1 — Ground-truth simulators

**User story:** As a researcher, I want simulators whose true energy is known exactly, so that I
can measure energy drift and out-of-distribution generalisation against ground truth.

1.1 The system SHALL provide four controlled environments: pendulum (torque), cart-pole (cart force),
    acrobot (elbow torque), planar 2-body orbit (2-D thrust).
1.2 Each environment SHALL expose canonical coordinates (q, p), the analytic Hamiltonian H(q, p),
    and a `reset(seed, energy_range)` that samples initial conditions whose energy lies in the
    requested band.
1.3 WHEN control is zero and damping is zero, THEN the simulator's relative energy drift over
    10,000 steps SHALL be < 1e-6 for pendulum and orbit and < 1e-5 for cart-pole and acrobot.
1.4 Each environment SHALL define a task reward and success predicate (pendulum swing-up,
    cart-pole swing-up, acrobot swing-up, orbit transfer to a target radius).
1.5 Simulators SHALL be deterministic given a seed.

### Requirement 2 — Datasets and splits

**User story:** As a researcher, I want reproducible datasets with energy-band splits, so that H2
tests real extrapolation instead of interpolation.

2.1 The system SHALL generate trajectory datasets with smooth random actions (Ornstein–Uhlenbeck),
    stored as `.npz` with states, canonical (q, p), actions and true energy; images are rendered
    from states at batch time rather than stored.
2.2 The system SHALL provide energy-band splits: train on a low band, test on in-band and on a
    disjoint higher band (`test_ood`).
2.3 Dataset generation SHALL be keyed by a config hash, so the same config never generates twice.

### Requirement 3 — Observations

3.1 The state observation SHALL be (generalised positions, generalised velocities), not canonical
    momenta; models that need p must infer it.
3.2 The system SHALL render 64×64 grayscale images of every environment without a display server.
3.3 Pixel models SHALL be given a short stack of frames (k = 3) so that velocity can be inferred.

### Requirement 4 — Baseline models

4.1 Model A: an MLP next-state predictor (residual Δ-state), action-conditioned.
4.2 Model B: a PINN — model A plus a residual loss against the known equations of motion
    (privileged knowledge, reported as such).
4.3 Model C: a latent Neural ODE with an RK4 integrator and an energy-consistency penalty, a
    faithful re-implementation of the friend's model extended with actions.
4.4 Model D: an RSSM (Dreamer-style: GRU deterministic state plus diagonal-Gaussian stochastic
    state, KL-balanced ELBO) for pixels and states.

### Requirement 5 — Structured model E (ours)

5.1 Model E SHALL map observations to a latent (q, p) of dimension 2n, evolve it with a learned
    Hamiltonian H_θ(q, p), an input matrix G_θ(q) u, and a positive semi-definite damping
    R_θ(q) ⪰ 0, and decode back to observations.
5.2 The conservative part SHALL be integrated with a symplectic integrator: leapfrog when H is
    separable, implicit midpoint otherwise. Control and damping SHALL be applied by Strang splitting.
5.3 WHEN u = 0 and R = 0, THEN the learned energy H_θ along a model rollout SHALL show bounded
    oscillation without secular drift (test: |H_θ(t) − H_θ(0)| at t = 10k is < 10× its value at t = 1k).
5.4 Model E-ens SHALL be an ensemble of 5 independently initialised and bootstrapped E models that
    exposes the mean prediction and the disagreement.

### Requirement 6 — Training

6.1 All models SHALL be trained with a multi-step rollout loss and a horizon curriculum, through
    one shared trainer driven by YAML configs.
6.2 Runs SHALL be seeded, write a checkpoint, the resolved config, and JSONL metrics to
    `results/<run_id>/`, and be resumable.
6.3 The trainer SHALL refuse a config whose model has ≥ 5M parameters.

### Requirement 7 — Evaluation (H1, H2)

7.1 The system SHALL report rollout error (normalised MSE in observation space) at 10, 100 and 1000 steps.
7.2 The system SHALL report energy drift over 10,000 free-evolution steps, measured with the **true**
    Hamiltonian applied to decoded states.
7.3 The system SHALL report every metric on the in-band and OOD splits.
7.4 Every headline number SHALL be aggregated over ≥ 3 seeds with 95% bootstrap confidence intervals.
7.5 The system SHALL evaluate H1 and H2 automatically against the thresholds in PREREGISTRATION.md
    and write a verdict table.

### Requirement 8 — Planning and control (H3)

8.1 The system SHALL provide a CEM planner in MPC mode that works with any model implementing the
    common interface.
8.2 WHEN planning with an ensemble, THEN the planner SHALL penalise ensemble disagreement with a
    configurable weight β.
8.3 The system SHALL run an iterative model-based RL loop (collect, train, plan) and log success
    rate and return against the cumulative number of environment steps.
8.4 The system SHALL measure calibration of ensemble disagreement against the real error
    (Spearman correlation and a binned reliability plot).
8.5 The system SHALL evaluate H3: the number of environment samples to reach 80% success, E-ens vs RSSM.

### Requirement 9 — Chaos stress test (H4)

9.1 The system SHALL repeat the H1–H3 protocol on the acrobot and report the advantage ratio
    (baseline error / E error) as a function of the rollout horizon and of the Lyapunov time.
9.2 The system SHALL estimate the acrobot's maximal Lyapunov exponent from the simulator.

### Requirement 10 — Reproducibility and packaging

10.1 The project SHALL be a uv-managed Python package `hwm` with an MIT LICENSE
     (Shushant Kumar Choudhary) declared in pyproject.toml and in the README.
10.2 PREREGISTRATION.md SHALL fix the H1–H4 thresholds and protocol before any model results exist,
     and be committed before the first training run.
10.3 One script SHALL regenerate every figure and table from `results/`.
10.4 README.md SHALL report results with CIs, and LIMITATIONS.md SHALL state what the results do not show.
10.5 The unit test suite SHALL run on CPU in < 2 minutes.
