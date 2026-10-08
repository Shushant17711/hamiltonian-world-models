# Implementation Plan — Hamiltonian World Models (hwm)

Build order follows PLAN.md's six phases: ground-truth physics first, then the shared contracts,
then baselines, then model E on state vectors (H1, H2), then pixels and RSSM, then planning (H3),
then the acrobot stress test (H4) and the write-up. Each experiment task consumes only code from
earlier tasks. "Done" = `results/verdicts.md` reports H1–H4 against the pre-registered thresholds,
`scripts/make_figures.py` regenerates every figure, and `uv run pytest` passes on CPU in < 2 min.

## Milestones

- After task 4: simulators, data and pre-registration frozen (phase 1).
- After task 9: H1/H2 verdicts on state vectors, a possible first public cut (phases 2–3).
- After task 13: H3 verdict (phases 4–5). After task 15: the full study (phase 6).

## Task list

- [x] 1. Scaffold the uv package, config system and test harness
  - `pyproject.toml` (uv, package `hwm`, deps torch/numpy/scipy/pyyaml/matplotlib, dev pytest/ruff),
    `license = "MIT"`, MIT `LICENSE` (Shushant Kumar Choudhary), README stub with License section, `.gitignore` (data/, results/* except verdicts/figures)
  - `hwm/config.py`: YAML → nested frozen dataclasses, `a.b=c` CLI overrides, `config_hash()`
  - `hwm/utils/seed.py` (seed torch/numpy/python); `tests/test_config.py` for round-trip, overrides and hash stability
  - _Requirements: 10.1, 6.2_

- [x] 2. Build the ground-truth simulators
- [x] 2.1 Implement the env contract, the GL4 integrator and the pendulum
  - `hwm/envs/base.py`: `Env` with `n, d_u, u_max, dt, E_ref`, `H`, `dH`, `obs_to_qp`, `qp_to_obs`, `step(qp, u)`, `sample_band(rng, band, N)`
  - `hwm/integrators/gl4.py`: vectorised 2-stage Gauss–Legendre, fixed-point to 1e-13, 10 substeps, control + damping in the vector field
  - `hwm/envs/pendulum.py`, `hwm/envs/registry.py` (`make(name)`)
  - Tests: dH vs finite differences, drift < 1e-6 over 10k steps, determinism, sampled energies inside band
  - _Requirements: 1.1, 1.2, 1.3, 1.5_

- [x] 2.2 Implement the orbit env with Kepler-element sampling
  - `hwm/envs/orbit.py`: H = ½‖p‖² − 1/‖q‖, 2-D thrust, `sample_band` over semi-major axis a with e ≤ 0.3
  - Tests: dH vs finite differences, drift < 1e-6 over 10k steps, sampled a and e recovered from (q, p), angular momentum conserved when u = 0
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 2.3 Implement cart-pole and acrobot (non-separable H)
  - `hwm/envs/cartpole.py`, `hwm/envs/acrobot.py` with analytic M(q), ∂M/∂q, V, dH; cart-pole |x| rejection and kinetic-share cap in `sample_band`
  - Torch twins of `qp_to_obs` / `accel(q, v, u)` per env (needed later by the PINN), tested equal to numpy
  - Tests: dH vs finite differences, drift < 1e-5 over 10k steps, (q, q̇) ↔ (q, p) round-trip
  - _Requirements: 1.1, 1.2, 1.3, 1.5_

- [x] 2.4 Add task rewards, success predicates and episode starts
  - `reward(obs, u)` (numpy + torch, batched), `success(obs_seq)`, `task_start()` per env per design §2
  - Tests: success is true on hand-built upright / transferred trajectories and false on hanging ones; numpy and torch rewards agree
  - _Requirements: 1.4_

- [x] 2.5 Implement the torch SDF renderer
  - `hwm/envs/render.py`: `render(env_name, obs, size=64) -> (…, 64, 64)` in [0, 1], segment/circle/box SDFs with smoothstep anti-aliasing, CPU and CUDA
  - Tests: output shape and range, pendulum frame centroid moves with θ, different states give different images, 1000 frames render in < 1 s on CPU
  - _Requirements: 3.2_

- [x] 3. Generate data and feed it to models
- [x] 3.1 Implement dataset generation with energy bands and OU actions
  - `hwm/data/generate.py`: OU actions, 25% passive, splits train/val/test_in/test_ood/test_long per design §3, `.npz` under `data/<env>/<hash>/`
  - `configs/data/{pendulum,cartpole,acrobot,orbit}.yaml`; `scripts/gen_data.py` skips work when the hash dir exists
  - Tests: tiny config generates every split with the right shapes, energies of OOD initial states outside the train band, same hash → identical files
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 3.2 Implement the window dataset, normalisation and pixel contexts
  - `hwm/data/dataset.py`: `WindowDataset(split, horizon, context, obs_mode)` returning `Batch(ctx, actions, target, rewards, passive)`; normaliser fit on train only
  - Pixel mode renders ctx/target on the fly via 2.5, with k = 3 context frames
  - Tests: window alignment (target[0] is the step after the last ctx frame), normaliser round-trip, pixel batch shapes
  - _Requirements: 3.1, 3.3, 2.1_

- [x] 4. Freeze the pre-registration in code and prose
  - `PREREGISTRATION.md` with H1–H4 protocol and thresholds copied from design §9, plus the metric definitions from §7
  - `hwm/eval/thresholds.py` with the same numbers as constants; a test parses PREREGISTRATION.md and asserts they match
  - Commit before any training code exists
  - _Requirements: 10.2, 7.5_

- [x] 5. Build the model contract, integrators and trainer
- [x] 5.1 Implement the WorldModel base, nets and torch integrators
  - `hwm/models/base.py` (`WorldModel`, `Rollout`, `Batch`), `hwm/models/nets.py` (MLP, state enc/dec, CNN enc/dec), `hwm/models/registry.py`
  - `hwm/integrators/torch_integrators.py`: `rk4`, `leapfrog`, `implicit_midpoint` (unrolled fixed-point, differentiable)
  - Tests: integrator orders on a harmonic oscillator (error slope), leapfrog/implicit-midpoint energy bounded over 10k steps, gradients flow through implicit midpoint
  - _Requirements: 5.2, 6.1_

- [x] 5.2 Implement the shared trainer with model A as its first client
  - `hwm/train/trainer.py`: AdamW + cosine, horizon curriculum, val early stopping, JSONL metrics, `ckpt.pt` / `ckpt_best.pt`, resume, < 5M-param assert
  - `hwm/models/mlp.py` (residual next-state MLP); `scripts/train.py --config … key=value`; `configs/model/mlp.yaml`
  - Tests: 50-step train on a tiny pendulum dataset lowers the loss, resume continues from the saved step, a 6M-param config is refused
  - _Requirements: 4.1, 6.1, 6.2, 6.3_

- [x] 6. Implement the remaining baselines
- [x] 6.1 Implement model B (PINN)
  - `hwm/models/pinn.py`: model A + physics residual on predicted states + 256 collocation points per batch, using the torch `accel` from 2.3; `configs/model/pinn.yaml`
  - Tests: residual is ~0 on true simulator transitions and > 0 on perturbed ones; contract test and tiny train
  - _Requirements: 4.2_

- [x] 6.2 Implement model C (latent Neural ODE + energy penalty)
  - `hwm/models/latent_ode.py`: encoder → z, `f_θ(z, u)` with RK4, energy head; passive-only variance penalty + energy supervision per design §5; `configs/model/latent_ode.yaml`
  - Tests: penalty is zero on non-passive batches, contract test, tiny train
  - _Requirements: 4.3_

- [x] 6.3 Implement model D (RSSM) in state mode
  - `hwm/models/rssm.py`: GRU 200 + Gaussian 30, posterior/prior, KL balancing 0.8, free nats 1; open-loop `rollout` via the prior; obs encoder/decoder pluggable for later pixel mode
  - Tests: KL ≥ free-nats floor, prior rollout shapes, tiny train
  - _Requirements: 4.4_

- [x] 7. Implement evaluation and statistics
- [x] 7.1 Implement rollout-error and energy-drift evaluation
  - `hwm/eval/rollout.py`: nMSE at h ∈ {10, 100, 1000} on test_in/test_ood/test_long, angles via (cos, sin)
  - `hwm/eval/energy.py`: 10k-step passive rollouts → decode → true H drift (and learned-energy drift when `energy()` is defined), chunked to fit 8 GB
  - `scripts/evaluate.py results/<run>` writes `results/<run>/eval/metrics.json`; tests on an oracle "model" that wraps the simulator (drift ≈ 0, nMSE ≈ 0)
  - _Requirements: 7.1, 7.2, 7.3_

- [x] 7.2 Implement bootstrap statistics and the sweep runner
  - `hwm/eval/stats.py`: `bootstrap_ci`, paired-difference CI, geometric-mean ratio CI
  - `scripts/sweep.py configs/sweeps/<name>.yaml`: env × model × seed grid → train + evaluate, skipping finished runs
  - Tests: CI covers the true mean in ≥ 90% of 200 synthetic trials; sweep dry-run lists the expected run ids
  - _Requirements: 7.4_

- [x] 8. Implement model E and its ensemble
- [x] 8.1 Implement the Hamiltonian world model (state mode)
  - `hwm/models/hamiltonian.py`: H_θ = ½pᵀA_θ(q)p + V_θ(q) with Cholesky A_θ, separable variant, angle features, G_θ(q), R_θ = KKᵀ, Strang step per design §5; `configs/model/hamiltonian.yaml`
  - Tests: JᵀΩJ = Ω (1e-4) for the conservative step, A_θ positive definite, learned energy bounded over 10k steps with u = R = 0 (the 1k vs 10k criterion), contract test, tiny train
  - _Requirements: 5.1, 5.2, 5.3_

- [x] 8.2 Implement the ensemble wrapper
  - `hwm/models/ensemble.py`: M members, per-member seeds and bootstrap trajectory resamples, batched `rollout` returning members/mean/disagreement; trainer support for training members in one run
  - Tests: members differ after init, disagreement is 0 when all members are identical copies
  - _Requirements: 5.4_

- [ ] 9. Run the state-vector study and evaluate H1/H2
- [x] 9.1 Implement the hypothesis evaluator for H1/H2
  - `hwm/eval/hypotheses.py`: load `results/**/eval/metrics.json`, apply `thresholds.py`, write `results/verdicts.md` with per-env tables and CIs
  - Tests on synthetic result trees that pass and fail each hypothesis
  - _Requirements: 7.5, 7.4_

- [ ] 9.2 Run the phase 2–3 sweep and commit verdicts
  - `configs/sweeps/state.yaml`: {pendulum, cartpole, acrobot, orbit} × {A, B, C, D, E} × 3 seeds; generate data, run sweep, evaluate, write H1/H2 verdicts
  - Reproduce the friend's setting (orbit, model C) as a sanity row and record that it matches the qualitative result
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 10. Extend to pixel observations
- [x] 10.1 Add pixel mode to model E
  - CNN encoder on 3 frames → (q, p), decoder from q only, reward head r(q, p, u); `configs/model/hamiltonian_pixels.yaml`
  - Tests: decoder signature ignores p, tiny pixel train on pendulum lowers loss, bf16 autocast only around CNNs
  - _Requirements: 3.3, 5.1_

- [x] 10.2 Add pixel mode to the RSSM
  - CNN encoder/decoder plugged into 6.3, reward head; `configs/model/rssm_pixels.yaml`
  - Tests: tiny pixel train lowers loss, peak GPU memory of a full-size batch < 7 GB (skipped without CUDA)
  - _Requirements: 4.4, 3.3_

- [ ] 11. Implement planning
- [x] 11.1 Implement the CEM planner
  - `hwm/planning/cem.py`: batched CEM with warm-start shift, momentum, action bounds; cost callback interface
  - Tests: finds the optimum of a known quadratic action cost; warm start shifts correctly
  - _Requirements: 8.1_

- [x] 11.2 Implement MPC over world models with a disagreement penalty
  - `hwm/planning/mpc.py`: encode context, roll out all candidates in one batch, reward from env (state) or head (pixels), `β·disagreement` for ensembles
  - Tests: with an oracle simulator model, MPC swings the pendulum up (success in 1 episode); β > 0 lowers the chosen plan's disagreement on a toy ensemble
  - _Requirements: 8.1, 8.2_

- [x] 11.3 Implement the model-based RL loop
  - `hwm/planning/mbrl.py` + `scripts/run_mbrl.py`: random warm-up, train/fine-tune → collect, eval at env-step budgets {1k, 2k, 5k, 10k, 20k, 50k}, `mbrl.jsonl`
  - Tests: 2-iteration smoke run with a tiny model writes well-formed records and appends data correctly
  - _Requirements: 8.3_

- [x] 11.4 Implement ensemble calibration analysis
  - `hwm/eval/calibration.py`: Spearman ρ of disagreement vs true error per horizon, 10-bin reliability curve; hook into `scripts/evaluate.py` for ensemble runs
  - Tests: perfectly correlated synthetic data gives ρ = 1, shuffled gives |ρ| < 0.1
  - _Requirements: 8.4_

- [ ] 12. Extend the hypothesis evaluator to H3
  - Steps-to-80%-success per run (∞ if never reached), E-ens/RSSM ratio per task, threshold from `thresholds.py`, appended to `verdicts.md`
  - Tests on synthetic `mbrl.jsonl` trees
  - _Requirements: 8.5_

- [ ] 13. Run the pixel + planning study and commit the H3 verdict
  - `configs/sweeps/mbrl.yaml`: {pendulum, cartpole, orbit} × {E-ens pixels, RSSM pixels, E-ens state} × 3 seeds; calibration eval on the E-ens runs
  - If wall-clock per seed exceeds the §12 budget, apply the documented population reduction and note it in the run config
  - _Requirements: 8.3, 8.4, 8.5_

- [ ] 14. Run the acrobot chaos stress test (H4)
- [x] 14.1 Implement the Lyapunov estimator and advantage-vs-horizon analysis
  - `hwm/eval/lyapunov.py` (Benettin on GL4) + `scripts/lyapunov.py`; `hypotheses.py` gains advantage(h / t_λ) with CIs and the H4 prediction check
  - Tests: λ ≈ 0 for the pendulum (|λ| < 0.05), λ > 0 for the acrobot at high energy
  - _Requirements: 9.1, 9.2_

- [ ] 14.2 Run the acrobot protocol and write the H4 verdict
  - Reuse the state-mode acrobot runs from 9.2, add acrobot to the MBRL sweep, run `scripts/lyapunov.py`, regenerate `verdicts.md`
  - _Requirements: 9.1_

- [ ] 15. Generate figures and write the report
  - `hwm/eval/figures.py` + `scripts/make_figures.py`: error vs horizon, energy drift curves, OOD bars, success vs env steps, reliability plot, advantage vs h/t_λ, all from `results/`
  - README.md: method, results tables with CIs pulled from `verdicts.md`, how to reproduce, license; LIMITATIONS.md: PINN privilege, latent-dim prior, compute reductions, what the results don't show
  - Test: `make_figures.py` runs on the synthetic result tree from 9.1
  - _Requirements: 10.3, 10.4, 10.5_

## Coverage

| Req | Tasks | Req | Tasks |
|---|---|---|---|
| 1.1–1.3, 1.5 | 2.1–2.3 | 6.1–6.3 | 5.1, 5.2 |
| 1.4 | 2.4 | 7.1–7.3 | 7.1, 9.2 |
| 2.1–2.3 | 3.1, 3.2 | 7.4 | 7.2, 9.1 |
| 3.1, 3.3 | 3.2, 10.1, 10.2 | 7.5 | 4, 9.1 |
| 3.2 | 2.5 | 8.1–8.2 | 11.1, 11.2 |
| 4.1–4.3 | 5.2, 6.1, 6.2 | 8.3–8.5 | 11.3, 11.4, 12, 13 |
| 4.4 | 6.3, 10.2 | 9.1–9.2 | 14.1, 14.2 |
| 5.1–5.3 | 5.1, 8.1, 10.1 | 10.1 | 1 |
| 5.4 | 8.2 | 10.2–10.5 | 4, 15 |
