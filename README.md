# hamiltonian-world-models

Action-conditioned world models whose structure (a learned Hamiltonian, port-Hamiltonian control and
damping, and a symplectic integrator) guarantees energy conservation in the absence of control. They
work from state vectors and from 64×64 pixels and are used for model-predictive control.

**Question.** Does building physics into a world model's *structure* beat learning it through a *loss*,
on long-horizon accuracy, generalisation to unseen energies, and sample-efficient control? And where does
the advantage stop on a chaotic system?

The hypotheses, metrics and decision rules were frozen in [`PREREGISTRATION.md`](PREREGISTRATION.md)
before any model code existed; [`results/verdicts.md`](results/verdicts.md) applies them mechanically.
Read [`LIMITATIONS.md`](LIMITATIONS.md) alongside the results.

## Method

| id | model | idea |
|---|---|---|
| A | MLP | residual next-state predictor |
| B | PINN* | A + residual against the true equations of motion (*privileged) |
| C | latent Neural ODE | RK4 latent dynamics + energy penalty: physics through the **loss** |
| D | RSSM | Dreamer-style recurrent state-space model |
| E | **Hamiltonian world model** | latent (q, p), H = ½ pᵀA(q)p + V(q) with A ≻ 0 by construction, input G(q)u and damping R(q) ⪰ 0, Strang splitting around a symplectic step: physics through the **structure** |
| E-ens | 5 × E | bootstrapped ensemble; its disagreement is penalised by the planner |

Four ground-truth systems with exactly known energy, simulated with a 4th-order Gauss–Legendre
integrator (relative energy drift < 1e-5 over 10k steps): pendulum (torque), cart-pole (force), acrobot
(elbow torque, chaotic) and the planar two-body orbit (thrust). Models train on a low energy band and are
tested in-band and on a disjoint higher band.

| hypothesis | pre-registered test |
|---|---|
| H1 | E's true-energy drift over 10k passive steps is ≥ 10× lower than C's (CI lower bound > 3) on ≥ 2 of 3 systems |
| H2 | E has lower out-of-band error at 100 steps than every baseline on ≥ 3 of 4 systems |
| H3 | with pixels, MPC with E-ens reaches 80% task success with ≤ ½ the env steps of the RSSM on ≥ 2 of 3 tasks |
| H4 | on the acrobot the advantage holds within one Lyapunov time and is gone beyond three |

## Results

Results are being produced; see [`results/verdicts.md`](results/verdicts.md) and `results/figures/`.

## Reproduce

```bash
uv sync
uv run pytest                                            # unit tests, CPU, < 2 min
uv run pytest -m slow                                    # 10k-step conservation checks

uv run python scripts/gen_data.py --config configs/data/{pendulum,cartpole,acrobot,orbit}.yaml
uv run python scripts/sweep.py configs/sweeps/state.yaml --shard 0/1   # H1, H2: 60 runs
uv run python scripts/lyapunov.py --env acrobot                         # H4
uv run python scripts/run_mbrl.py --config configs/model/ensemble.yaml env=pendulum obs_mode=pixels  # H3
uv run python scripts/make_figures.py                                   # verdicts.md + figures
```

One run: `uv run python scripts/train.py --config configs/model/hamiltonian.yaml env=acrobot seed=1`, then
`uv run python scripts/evaluate.py results/acrobot-hamiltonian-state-s1`. Every run writes its resolved
config, metrics and checkpoints under `results/<env>-<model>-<obs>-s<seed>/` and resumes if interrupted.
Hardware: one RTX 4060 Laptop GPU (8 GB); every model is under 5M parameters.

## Layout

```
hwm/envs         simulators, rewards, success predicates, 64×64 SDF renderer
hwm/integrators  GL4 (ground truth), rk4 / leapfrog / implicit midpoint (models)
hwm/data         dataset generation (energy bands, OU actions), training windows
hwm/models       A-E, ensemble, shared contract
hwm/train        the shared trainer
hwm/planning     CEM, MPC, model-based RL loop
hwm/eval         rollout error, energy drift, statistics, calibration, Lyapunov, verdicts, figures
scripts/         gen_data, train, evaluate, sweep, run_mbrl, lyapunov, make_figures
```

## License

MIT. See [LICENSE](LICENSE). Copyright (c) 2026 Shushant Kumar Choudhary.
