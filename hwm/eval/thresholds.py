"""Pre-registered H1–H4 thresholds (PREREGISTRATION.md §5, design §9).

These constants are the only copy the evaluation code may read. `tests/test_preregistration.py`
asserts that they equal the YAML block in PREREGISTRATION.md and that the frozen part of that file
(everything above "## Amendments") still has the hash below. Changing a number here without an
amendment in PREREGISTRATION.md fails the test suite on purpose.
"""

from typing import Any

# sha256 of PREREGISTRATION.md up to (not including) the "## Amendments" heading.
PREREGISTRATION_SHA256 = "c12b2c08c2cf7839d838371e165ac748d8e27ccdd22430b799d43e5071a5b873"

SEEDS = (0, 1, 2)

BOOTSTRAP_N = 10_000
BOOTSTRAP_ALPHA = 0.05
BOOTSTRAP_RNG_SEED = 0

NMSE_CAP = 100.0  # per-trajectory nMSE cap; non-finite errors count as the cap
DRIFT_FLOOR = 1e-9  # drift values are floored before any ratio is taken
DRIFT_STEPS = 10_000
DRIFT_INITIAL_STATES = 50
ROLLOUT_HORIZONS = (10, 100, 1000)

# H1: energy drift, state mode. Pass per env: geo-mean ratio drift(C)/drift(E) >= MIN_RATIO and CI lower > MIN_CI_LOWER.
H1_ENVS = ("pendulum", "cartpole", "orbit")
H1_OBS_MODE = "state"
H1_BAND = "test_in"
H1_MIN_RATIO = 10.0
H1_MIN_CI_LOWER = 3.0
H1_MIN_ENVS_PASSING = 2

# H2: test_ood nMSE at h = 100, E beats every baseline (paired-difference CI above 0).
H2_ENVS = ("pendulum", "cartpole", "acrobot", "orbit")
H2_OBS_MODE = "state"
H2_SPLIT = "test_ood"
H2_HORIZON = 100
H2_BASELINES = ("A", "B", "C", "D")
H2_MIN_ENVS_PASSING = 3

# H3: pixel-mode MBRL, median-over-seeds N80(E-ens) <= MAX_RATIO * N80(RSSM).
H3_ENVS = ("pendulum", "cartpole", "orbit")
H3_OBS_MODE = "pixels"
H3_SUCCESS_THRESHOLD = 0.8
H3_EVAL_EPISODES = 10
H3_CHECKPOINTS = (1_000, 2_000, 5_000, 10_000, 20_000, 50_000)
H3_MAX_RATIO = 0.5
H3_BETA_ENSEMBLE = 1.0
H3_MIN_ENVS_PASSING = 2

# H4: acrobot advantage(h) vs h*dt / t_lambda (descriptive, with a stated prediction).
H4_ENV = "acrobot"
H4_HORIZONS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
H4_EARLY_MAX_LYAPUNOV_TIMES = 1.0  # advantage CI above 1 for every h*dt below this many t_lambda
H4_LATE_MIN_LYAPUNOV_TIMES = 3.0  # advantage CI includes 1 for every h*dt above this many t_lambda
H4_LYAPUNOV_D0 = 1e-8
H4_LYAPUNOV_INITIAL_STATES = 20


THRESHOLDS: dict[str, Any] = {
    "seeds": list(SEEDS),
    "bootstrap": {"n_resamples": BOOTSTRAP_N, "alpha": BOOTSTRAP_ALPHA, "rng_seed": BOOTSTRAP_RNG_SEED},
    "metrics": {
        "nmse_cap": NMSE_CAP,
        "drift_floor": DRIFT_FLOOR,
        "drift_steps": DRIFT_STEPS,
        "drift_initial_states": DRIFT_INITIAL_STATES,
        "rollout_horizons": list(ROLLOUT_HORIZONS),
    },
    "H1": {
        "envs": list(H1_ENVS),
        "obs_mode": H1_OBS_MODE,
        "band": H1_BAND,
        "min_ratio": H1_MIN_RATIO,
        "min_ci_lower": H1_MIN_CI_LOWER,
        "min_envs_passing": H1_MIN_ENVS_PASSING,
    },
    "H2": {
        "envs": list(H2_ENVS),
        "obs_mode": H2_OBS_MODE,
        "split": H2_SPLIT,
        "horizon": H2_HORIZON,
        "baselines": list(H2_BASELINES),
        "min_envs_passing": H2_MIN_ENVS_PASSING,
    },
    "H3": {
        "envs": list(H3_ENVS),
        "obs_mode": H3_OBS_MODE,
        "success_threshold": H3_SUCCESS_THRESHOLD,
        "eval_episodes": H3_EVAL_EPISODES,
        "checkpoints": list(H3_CHECKPOINTS),
        "max_ratio": H3_MAX_RATIO,
        "beta_ensemble": H3_BETA_ENSEMBLE,
        "min_envs_passing": H3_MIN_ENVS_PASSING,
    },
    "H4": {
        "env": H4_ENV,
        "horizons": list(H4_HORIZONS),
        "early_max_lyapunov_times": H4_EARLY_MAX_LYAPUNOV_TIMES,
        "late_min_lyapunov_times": H4_LATE_MIN_LYAPUNOV_TIMES,
        "lyapunov_d0": H4_LYAPUNOV_D0,
        "lyapunov_initial_states": H4_LYAPUNOV_INITIAL_STATES,
    },
}
