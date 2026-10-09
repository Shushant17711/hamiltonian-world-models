# Limitations

What the results in this repository do **not** show, and the choices that could bias them. Read this next
to `results/verdicts.md`.

## Comparisons that are not like for like

- **Model B (PINN) is privileged.** Its physics residual uses the simulator's true equations of motion
  (`env.accel`). No other model gets that, and every table marks B with an asterisk. A PINN that beats E
  says that knowing the dynamics helps, not that a loss beats structure.
- **Model E gets two structural priors that the baselines do not:** its latent has exactly 2n
  coordinates (n = the system's true degrees of freedom), and the latent coordinates that correspond to
  angles are fed to its networks through (cos, sin). Model C's latent also has dimension 2n, but it
  does not get the angle features. Part of any advantage of E may come from these priors rather than
  from the symplectic structure itself. The n_lat = 2n ablation (exploratory) is the first check on that.
- **E is torch.compile'd on CUDA; the other models are not.** This changes speed only, not the maths,
  and it is why E's training is affordable. Wall-clock numbers are not comparable across models.
- **Model E received stability fixes after its first runs diverged** (PREREGISTRATION.md, amendment 1):
  four diagnostic configurations on the validation split, chosen to make training stable on all four
  systems. The baselines trained stably with the defaults and received none. This favours E.
- **Hyperparameters were not otherwise tuned for any model.** All models use the shared trainer defaults of
  design §6 (PREREGISTRATION.md §2). A baseline with a better learning rate or width might do better.

## What the metrics measure

- **Energy drift is measured through the decoder.** The true Hamiltonian is evaluated on decoded states,
  so a model with perfectly conserved latent energy still shows the decoder's error. That is intended
  (it is the energy of what the model predicts) but it puts a floor under every model's drift.
- **Three seeds.** With 3 seeds the lower end of a 95% percentile-bootstrap interval is the worst seed
  (PREREGISTRATION.md §4). The intervals describe seed-to-seed variation only; every seed of a given env
  uses the same generated dataset (data seed 0), so dataset variation is not covered.
- **One dataset size and one energy split per system.** The out-of-distribution test is a single higher
  energy band; it says nothing about other kinds of shift (parameters, damping, observation noise).
- **The cart-pole pole is ~5 px long in the 64x64 images**, because the fixed camera must cover the whole
  track. Pixel results on the cart-pole are therefore hard for every model.

## Reading the H1/H2/H4 results

- **"Not supported" is a statement about this implementation at this budget** (30k steps, untuned
  defaults, 3 seeds), not about Hamiltonian structure in general. E's failures are dominated by seeds whose
  learned Hamiltonian was wrong away from the data; a different training budget, a better-conditioned
  latent, or more seeds could change the verdicts. They were not tried, by design.
- **Several baselines diverge in long passive rollouts** (C on the pendulum and cart-pole, B and A in places),
  which makes their drift infinite on some seeds. The ratio CIs then have infinite upper ends; the
  verdicts depend on the lower ends only.
- **The orbit results expose a gap in the solver fix**: one orbit seed did not even conserve its learned
  energy (learned-H drift ~1e3), meaning 8 adaptive substeps were not enough there.

## Planning (H3)

- **Pixel-mode ensemble disagreement uses the members' predicted rewards**, not decoded frames
  (design §8): decoding 64x64 frames for every CEM candidate was unaffordable. State-mode disagreement
  is the std of decoded states as specified.
- **Runs stop at the first checkpoint that reaches 80% success** (`stop_at_success`). This cannot change
  N80 or the H3 verdict, but the success-vs-steps curves end there.
- Compute-driven changes to the MBRL protocol, if any, are listed under Amendments in
  `PREREGISTRATION.md` with the reason; the verdict under the original rule is reported first wherever
  it can be computed.
