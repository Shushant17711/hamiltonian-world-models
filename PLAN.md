# Hamiltonian World Models — plan (handoff)

Status: planning only, no code yet. Decisions still open are at the bottom.

## Pitch
An action-conditioned, pixel-capable world model whose structure guarantees energy conservation,
used for planning and control.

## Motivation
Improves on Optimus2007/physics-informed-world-models (2-body orbit; MLP vs PINN vs latent Neural ODE
with an energy penalty; latent_dim=4, hidden_dim=64, ~700 lines). Gaps there: no actions (a predictor,
not a world model), energy only encouraged by the loss, state vectors only, one system and one
trajectory, no out-of-distribution test, no uncertainty, no downstream task.

## Core question
Does building physics into the model's structure (Hamiltonian + symplectic integrator) beat learning
it through a loss on long-horizon accuracy, generalisation to new conditions, and control?

## Pre-registered hypotheses
- H1: the structured model's energy drift over 10k steps is >=10x lower than the energy-penalised Neural ODE.
- H2: lower error than all baselines on unseen energies when trained on a limited energy range.
- H3: MPC with the structured model reaches task success with fewer environment samples than an RSSM (Dreamer-style).
- H4: on the chaotic acrobot the advantage shrinks; measure where it holds and where it breaks.

## Environments (own simulators, true energy known exactly)
pendulum (torque, swing-up) · cart-pole (force) · acrobot (elbow torque, chaotic) · 2-body orbit (thrust).
Observations: clean state vectors, and rendered 64x64 images.

## Models
A plain MLP next-state · B PINN · C latent Neural ODE + energy penalty (re-implementation of the friend's model)
· D RSSM · E (ours) encoder -> latent (q,p) -> Hamiltonian net + control term + damping -> leapfrog -> decoder
· E-ens: 5-model ensemble for uncertainty.

## Planning
CEM/MPC in latent space; ensemble disagreement is penalised so the planner can't exploit model errors.

## Metrics
Rollout error at 10/100/1000 steps, energy drift over 10k steps, error on unseen energies and
initial conditions, success rate and reward vs number of environment samples, calibration of
ensemble disagreement against real error, >=3 seeds with confidence intervals.

## Layout
hwm/{envs,models,integrators,planning,eval}, scripts/, configs/ (YAML), tests/, results/,
README.md, LICENSE (MIT, Shushant Kumar Choudhary), LIMITATIONS.md, PREREGISTRATION.md, pyproject.toml (uv).

## Constraints
RTX 4060 Laptop, 8 GB VRAM -> models under ~5M parameters, 64x64 images. Python + uv, PyTorch, matplotlib.
No Claude co-author line in commits.

## Phases
1. Simulators, integrators and tests (verify true energy conservation)
2. Baselines A-C on state vectors (reproduce the friend's result)
3. Model E on state vectors -> test H1, H2
4. Pixel observations + RSSM baseline
5. Planning with the ensemble -> test H3
6. Acrobot stress test -> H4; figures, README, report

## Open decisions
1. Final name (working name: hamiltonian-world-models)
2. Scope: all 6 phases, or a first public version after phase 3
3. Workflow: tasks.md with subagents, or phase-by-phase
