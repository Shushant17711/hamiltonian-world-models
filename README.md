# hamiltonian-world-models

Action-conditioned world models whose structure (a learned Hamiltonian, port-Hamiltonian control and
damping, and a symplectic integrator) guarantees energy conservation in the absence of control. They
work from state vectors and from 64×64 pixels and are used for model-predictive control.

Status: under construction. Spec in `.kiro/specs/hamiltonian-world-models/`, original plan in `PLAN.md`.

## Setup

```bash
uv sync
uv run pytest
```

## License

MIT. See [LICENSE](LICENSE). Copyright (c) 2026 Shushant Kumar Choudhary.
