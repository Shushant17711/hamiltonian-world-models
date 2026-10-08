"""PREREGISTRATION.md and hwm.eval.thresholds must agree, and the frozen protocol must not change (Req 10.2, 7.5)."""

import hashlib
import re
from pathlib import Path

import yaml

from hwm.envs.registry import names as env_names
from hwm.eval import thresholds

PREREG = Path(__file__).resolve().parents[1] / "PREREGISTRATION.md"
AMENDMENTS_HEADING = "\n## Amendments"


def _text() -> str:
    return PREREG.read_text(encoding="utf-8")


def frozen_part(text: str) -> str:
    idx = text.find(AMENDMENTS_HEADING)
    assert idx >= 0, "PREREGISTRATION.md must keep an '## Amendments' section"
    return text[:idx]


def _yaml_block(text: str) -> dict:
    blocks = re.findall(r"```yaml\n(.*?)```", text, flags=re.DOTALL)
    assert len(blocks) == 1, f"expected exactly one yaml block, found {len(blocks)}"
    return yaml.safe_load(blocks[0])


def test_yaml_block_matches_constants():
    assert _yaml_block(_text()) == thresholds.THRESHOLDS


def test_frozen_protocol_hash_unchanged():
    digest = hashlib.sha256(frozen_part(_text()).encode("utf-8")).hexdigest()
    assert digest == thresholds.PREREGISTRATION_SHA256, (
        "The frozen part of PREREGISTRATION.md changed. Record the change under '## Amendments' "
        "instead, or, if this is a deliberate amendment, update PREREGISTRATION_SHA256."
    )


def test_prose_states_the_headline_numbers():
    prose = frozen_part(_text()).split("## 7. Machine-readable copy")[0]
    for needle in [
        "geometric mean of r_s over seeds is ≥ 10",
        "lower bound\nof its 95% bootstrap CI is > 3",
        "≥ 2 of the 3 envs pass",
        "`test_ood` at h = 100",
        "≥ 3 of the 4 envs pass",
        "N80(E-ens) ≤ 0.5 × N80(RSSM)",
        "≥ 2 of the 3 tasks pass",
        "h·dt < 1 t_λ",
        "h·dt > 3 t_λ",
    ]:
        assert needle in prose, f"prose no longer states: {needle!r}"


def test_env_names_and_internal_consistency():
    known = set(env_names())
    t = thresholds
    for envs in (t.H1_ENVS, t.H2_ENVS, t.H3_ENVS, (t.H4_ENV,)):
        assert set(envs) <= known
    assert t.H4_ENV not in t.H1_ENVS  # acrobot is reserved for H4
    assert t.H1_MIN_ENVS_PASSING <= len(t.H1_ENVS)
    assert t.H2_MIN_ENVS_PASSING <= len(t.H2_ENVS)
    assert t.H3_MIN_ENVS_PASSING <= len(t.H3_ENVS)
    assert t.H1_MIN_CI_LOWER < t.H1_MIN_RATIO
    assert t.H2_HORIZON in t.ROLLOUT_HORIZONS
    assert t.H4_EARLY_MAX_LYAPUNOV_TIMES < t.H4_LATE_MIN_LYAPUNOV_TIMES
    assert list(t.H3_CHECKPOINTS) == sorted(t.H3_CHECKPOINTS)
    assert max(t.H4_HORIZONS) <= max(t.ROLLOUT_HORIZONS)
