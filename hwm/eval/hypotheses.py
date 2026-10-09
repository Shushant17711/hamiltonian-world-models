"""Mechanical H1-H4 verdicts from evaluated runs (Req 7.5, 8.5, 9.1; PREREGISTRATION.md §5).

Reads every ``<results>/**/eval/metrics.json``, keeps the pre-registered seeds, applies the rules in
``hwm.eval.thresholds`` and writes ``verdicts.md`` (and ``verdicts.json``). Nothing here is tunable:
every number comes from ``thresholds``. Runs with other seeds are listed as exploratory and ignored.

An env whose required (model, seed) runs are not all present gets the status ``incomplete``, and a
hypothesis with an incomplete env is ``incomplete`` unless the complete envs already decide it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from hwm.eval import thresholds as T
from hwm.eval.stats import CI, geo_mean_ratio_ci, paired_difference_ci

# model letters of PREREGISTRATION.md §2 -> registry names
MODEL_IDS = {"A": "mlp", "B": "pinn", "C": "latent_ode", "D": "rssm", "E": "hamiltonian"}
PRIVILEGED = {"B"}


def load_metrics(results: str | Path) -> list[dict]:
    out = []
    for p in sorted(Path(results).glob("**/eval/metrics.json")):
        m = json.loads(p.read_text())
        m["_path"] = str(p)
        out.append(m)
    return out


def index(metrics: Iterable[dict]) -> dict[tuple[str, str, str, int], dict]:
    """(env, model name, obs_mode, seed) -> metrics."""
    return {(m["env"], m["model"], m.get("obs_mode", "state"), int(m["seed"])): m for m in metrics}


@dataclass
class EnvResult:
    env: str
    status: str  # pass | fail | incomplete
    detail: dict = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)


@dataclass
class Verdict:
    hypothesis: str
    status: str  # supported | not supported | incomplete
    envs: list[EnvResult]
    rule: str


def _collect(idx, env: str, model: str, obs: str, getter) -> tuple[list[float] | None, list[str]]:
    vals, missing = [], []
    for s in T.SEEDS:
        m = idx.get((env, MODEL_IDS[model], obs, s))
        if m is None:
            missing.append(f"{env}-{MODEL_IDS[model]}-{obs}-s{s}")
            continue
        vals.append(float(getter(m)))
    return (vals if not missing else None), missing


def _decide(name: str, envs: list[EnvResult], need: int, rule: str) -> Verdict:
    n_pass = sum(e.status == "pass" for e in envs)
    n_open = sum(e.status == "incomplete" for e in envs)
    if n_pass >= need:
        status = "supported"
    elif n_pass + n_open < need:
        status = "not supported"
    else:
        status = "incomplete"
    return Verdict(name, status, envs, rule)


def evaluate_h1(idx) -> Verdict:
    envs = []
    for env in T.H1_ENVS:

        def drift(m):
            return m["drift"][T.H1_BAND]["median"]

        c, miss_c = _collect(idx, env, "C", T.H1_OBS_MODE, drift)
        e, miss_e = _collect(idx, env, "E", T.H1_OBS_MODE, drift)
        if c is None or e is None:
            envs.append(EnvResult(env, "incomplete", missing=miss_c + miss_e))
            continue
        r = geo_mean_ratio_ci(c, e)
        ok = r.mean >= T.H1_MIN_RATIO and r.lo > T.H1_MIN_CI_LOWER
        envs.append(EnvResult(env, "pass" if ok else "fail", {"ratio": r, "drift_C": c, "drift_E": e}))
    rule = (
        f"per env: geometric-mean drift(C)/drift(E) >= {T.H1_MIN_RATIO:g} and 95% CI lower bound > "
        f"{T.H1_MIN_CI_LOWER:g}; supported if >= {T.H1_MIN_ENVS_PASSING} of {len(T.H1_ENVS)} envs pass"
    )
    return _decide("H1", envs, T.H1_MIN_ENVS_PASSING, rule)


def evaluate_h2(idx) -> Verdict:
    envs = []
    h = str(T.H2_HORIZON)
    for env in T.H2_ENVS:

        def nmse(m):
            return m["nmse"][T.H2_SPLIT][h]

        e, missing = _collect(idx, env, "E", T.H2_OBS_MODE, nmse)
        diffs, beaten = {}, {}
        for x in T.H2_BASELINES:
            v, miss = _collect(idx, env, x, T.H2_OBS_MODE, nmse)
            missing += miss
            if v is not None and e is not None:
                diffs[x] = paired_difference_ci(v, e)
                beaten[x] = diffs[x].lo > 0
        if missing:
            envs.append(EnvResult(env, "incomplete", {"diff": diffs}, missing))
            continue
        status = "pass" if all(beaten.values()) else "fail"
        envs.append(EnvResult(env, status, {"diff": diffs, "beaten": beaten, "nmse_E": e}))
    rule = (
        f"per env: on {T.H2_SPLIT} at h = {T.H2_HORIZON}, the 95% CI of mean_s(nMSE_X - nMSE_E) lies above 0 "
        f"for every X in {{{', '.join(T.H2_BASELINES)}}}; supported if >= {T.H2_MIN_ENVS_PASSING} of "
        f"{len(T.H2_ENVS)} envs pass"
    )
    return _decide("H2", envs, T.H2_MIN_ENVS_PASSING, rule)


def _advantage_rows(idx, split: str, horizons, dt: float, t_lambda: float) -> tuple[list[dict], list[str]]:
    rows, missing = [], []
    env, obs = T.H4_ENV, "state"

    def curve(h):
        return lambda m: m["nmse_curve"][split][str(h)]

    for h in horizons:
        e, miss = _collect(idx, env, "E", obs, curve(h))
        base, means = {}, {}
        for x in T.H2_BASELINES:
            v, mx = _collect(idx, env, x, obs, curve(h))
            miss += mx
            if v is not None:
                base[x], means[x] = v, sum(v) / len(v)
        if miss or e is None:
            missing += miss
            continue
        best = min(means, key=means.get)  # the strongest baseline at this horizon
        ci = geo_mean_ratio_ci(base[best], e, floor=1e-12)
        rows.append(
            {
                "h": h,
                "x": h * dt / t_lambda,
                "baseline": best,
                "advantage": ci,
                "per_baseline": {x: geo_mean_ratio_ci(base[x], e, floor=1e-12) for x in base},
            }
        )
    return rows, sorted(set(missing))


def _prediction(rows: list[dict]) -> tuple[str, str, str]:
    early = [r for r in rows if r["x"] < T.H4_EARLY_MAX_LYAPUNOV_TIMES]
    late = [r for r in rows if r["x"] > T.H4_LATE_MIN_LYAPUNOV_TIMES]
    a = "untestable" if not early else ("holds" if all(r["advantage"].lo > 1 for r in early) else "fails")
    b = (
        "untestable"
        if not late
        else ("holds" if all(r["advantage"].lo <= 1 <= r["advantage"].hi for r in late) else "fails")
    )
    tested = [p for p in (a, b) if p != "untestable"]
    if not tested:
        status = "untestable"
    elif all(p == "holds" for p in tested):
        status = "confirmed" if len(tested) == 2 else "partially confirmed"
    else:
        status = "partially confirmed" if "holds" in tested else "not confirmed"
    return status, a, b


def evaluate_h4(idx, lyap: dict | None) -> Verdict:
    rule = (
        f"advantage(h) = nMSE_baseline(h) / nMSE_E(h) on {T.H4_ENV}, baseline = the strongest of "
        f"{{{', '.join(T.H2_BASELINES)}}} at each h, 95% CI by paired log-ratio bootstrap; prediction "
        f"(a) CI above 1 for every h·dt < {T.H4_EARLY_MAX_LYAPUNOV_TIMES:g} t_λ, (b) CI includes 1 for every "
        f"h·dt > {T.H4_LATE_MIN_LYAPUNOV_TIMES:g} t_λ; confirmed if both hold, partially if one"
    )
    if lyap is None:
        return Verdict(
            "H4", "incomplete", [EnvResult(T.H4_ENV, "incomplete", missing=["lyapunov_acrobot.json"])], rule
        )
    dt, t_lam = lyap["dt"], lyap["bands"]["test_in"]["t_lambda"]
    primary, missing = _advantage_rows(idx, "test_long", T.H4_HORIZONS, dt, t_lam)
    secondary, _ = _advantage_rows(idx, "test_in", [h for h in T.H4_HORIZONS if h <= 100], dt, t_lam)
    if missing:
        return Verdict("H4", "incomplete", [EnvResult(T.H4_ENV, "incomplete", missing=missing)], rule)
    status, a, b = _prediction(primary)

    # the H1/H2 metrics on the acrobot, reported without thresholds
    def drift(m):
        return m["drift"][T.H1_BAND]["median"]

    c, _ = _collect(idx, T.H4_ENV, "C", "state", drift)
    e, _ = _collect(idx, T.H4_ENV, "E", "state", drift)
    detail = {
        "t_lambda": t_lam,
        "lambda": lyap["bands"]["test_in"]["lambda"],
        "dt": dt,
        "rows": primary,
        "rows_actuated": secondary,
        "part_a": a,
        "part_b": b,
        "drift_ratio_C_over_E": geo_mean_ratio_ci(c, e) if c and e else None,
    }
    return Verdict("H4", status, [EnvResult(T.H4_ENV, status, detail)], rule)


# --- H3 -------------------------------------------------------------------------------------------------
H3_MODELS = {"E-ens": "hamiltonian_ens", "RSSM": "rssm"}


def load_mbrl(results: str | Path) -> dict[str, list[dict]]:
    """run name -> mbrl.jsonl records, for every ``mbrl-*`` run directory."""
    out = {}
    for p in sorted(Path(results).glob("mbrl-*/mbrl.jsonl")):
        out[p.parent.name] = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return out


def n80(records: list[dict]) -> float | None:
    """Env steps (checkpoint label) to the first >= threshold success, inf if never within the budget,
    None while undetermined (the run has neither succeeded nor evaluated every checkpoint)."""
    done = set()
    for r in sorted(records, key=lambda r: r["checkpoint"]):
        if r["success_rate"] >= T.H3_SUCCESS_THRESHOLD:
            return float(r["checkpoint"])
        done.add(r["checkpoint"])
    return math.inf if set(T.H3_CHECKPOINTS_AMENDED) <= done else None


def _median(v: list[float]) -> float:
    v = sorted(v)
    return v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2])


def evaluate_h3(runs: dict[str, list[dict]]) -> Verdict:
    rule = (
        f"per task (pixel mode): median over seeds of N80 = env steps to the first checkpoint with success >= "
        f"{T.H3_SUCCESS_THRESHOLD:.0%} ({T.H3_EVAL_EPISODES} MPC episodes; inf if never within "
        f"{max(T.H3_CHECKPOINTS_AMENDED):,} env steps, amendment 2); pass if N80(E-ens) <= {T.H3_MAX_RATIO:g} x N80(RSSM) (finite <= inf passes, "
        f"inf vs inf fails); supported if >= {T.H3_MIN_ENVS_PASSING} of {len(T.H3_ENVS)} tasks pass"
    )
    envs = []
    for env in T.H3_ENVS:
        vals, missing = {}, []
        for label, name in H3_MODELS.items():
            per_seed = []
            for s in T.SEEDS:
                rid = f"mbrl-{env}-{name}-{T.H3_OBS_MODE}-s{s}"
                v = n80(runs.get(rid, []))
                if v is None:
                    missing.append(rid)
                else:
                    per_seed.append(v)
            vals[label] = per_seed
        if missing:
            envs.append(EnvResult(env, "incomplete", {"n80": vals}, missing))
            continue
        e, r = _median(vals["E-ens"]), _median(vals["RSSM"])
        ok = math.isfinite(e) and (math.isinf(r) or e <= T.H3_MAX_RATIO * r)
        envs.append(EnvResult(env, "pass" if ok else "fail", {"n80": vals, "median_E": e, "median_RSSM": r}))
    return _decide("H3", envs, T.H3_MIN_ENVS_PASSING, rule)


# --- report ------------------------------------------------------------------------------------------
def _fmt(x: float) -> str:
    if math.isinf(x):
        return "∞" if x > 0 else "-∞"
    if math.isnan(x):
        return "nan"
    return f"{x:.3g}"


def _ci(c: CI) -> str:
    return f"{_fmt(c.mean)} [{_fmt(c.lo)}, {_fmt(c.hi)}]"


def _seeds(v: list[float]) -> str:
    return ", ".join(_fmt(x) for x in v)


def render(verdicts: list[Verdict], exploratory: list[str]) -> str:
    lines = [
        "# Verdicts",
        "",
        "Generated by `hwm/eval/hypotheses.py` from `results/**/eval/metrics.json` under the rules frozen in",
        (
            f"`PREREGISTRATION.md` (seeds {list(T.SEEDS)}; 95% percentile bootstrap over seeds, "
            f"{T.BOOTSTRAP_N:,} resamples). Do not edit by hand."
        ),
        "",
        "| hypothesis | verdict |",
        "|---|---|",
        *[f"| {v.hypothesis} | **{v.status}** |" for v in verdicts],
        "",
    ]
    for v in verdicts:
        lines += [f"## {v.hypothesis}: {v.status}", "", f"Rule: {v.rule}.", ""]
        if v.hypothesis == "H1":
            lines += [
                "| env | drift(C) per seed | drift(E) per seed | ratio C/E [95% CI] | result |",
                "|---|---|---|---|---|",
            ]
            for e in v.envs:
                if e.status == "incomplete":
                    lines.append(f"| {e.env} | | | | incomplete (missing {len(e.missing)} runs) |")
                    continue
                d = e.detail
                lines.append(
                    f"| {e.env} | {_seeds(d['drift_C'])} | {_seeds(d['drift_E'])} | {_ci(d['ratio'])} | {e.status} |"
                )
            lines.append("")
            lines.append(
                f"Drift: median over {T.DRIFT_INITIAL_STATES} train-band initial states of "
                f"max_t |H(x_t) - H(x_0)| / E_ref over {T.DRIFT_STEPS:,} passive steps (true H)."
            )
        elif v.hypothesis == "H4":
            lines += _render_h4(v)
        elif v.hypothesis == "H3":
            lines += [
                "| task | N80 E-ens per seed | N80 RSSM per seed | median E-ens / RSSM | result |",
                "|---|---|---|---|---|",
            ]
            for e in v.envs:
                d = e.detail
                if e.status == "incomplete":
                    lines.append(f"| {e.env} | | | | incomplete (missing {len(e.missing)} runs) |")
                    continue
                lines.append(
                    f"| {e.env} | {_seeds(d['n80']['E-ens'])} | {_seeds(d['n80']['RSSM'])} | "
                    f"{_fmt(d['median_E'])} / {_fmt(d['median_RSSM'])} | {e.status} |"
                )
        else:
            head = " | ".join(f"{x} − E{' (privileged)' if x in PRIVILEGED else ''}" for x in T.H2_BASELINES)
            lines += [
                f"| env | nMSE(E) per seed | {head} | result |",
                "|---|---|" + "---|" * len(T.H2_BASELINES) + "---|",
            ]
            for e in v.envs:
                if e.status == "incomplete":
                    lines.append(
                        f"| {e.env} | |"
                        + " |" * len(T.H2_BASELINES)
                        + f" incomplete (missing {len(e.missing)} runs) |"
                    )
                    continue
                d = e.detail
                cells = " | ".join(
                    _ci(d["diff"][x]) + ("" if d["beaten"][x] else " ✗") for x in T.H2_BASELINES
                )
                lines.append(f"| {e.env} | {_seeds(d['nmse_E'])} | {cells} | {e.status} |")
            lines.append("")
            lines.append(
                "Cells: mean over seeds of nMSE_X − nMSE_E [95% CI]; ✗ marks a baseline E does not beat."
            )
        missing = sorted({m for e in v.envs for m in e.missing})
        if missing:
            lines += ["", "Missing runs: " + ", ".join(f"`{m}`" for m in missing)]
        lines.append("")
    if exploratory:
        lines += [
            "## Exploratory runs (not used in any verdict)",
            "",
            *[f"- `{r}`" for r in sorted(exploratory)],
            "",
        ]
    return "\n".join(lines)


def _render_h4(v: Verdict) -> list[str]:
    e = v.envs[0]
    if e.status == "incomplete":
        return []
    d = e.detail
    out = [
        (
            f"Maximal Lyapunov exponent (train band, GL4 simulator): λ = {d['lambda']:.3g} /s, "
            f"t_λ = {d['t_lambda']:.3g} s = {d['t_lambda'] / d['dt']:.0f} steps."
        ),
        "",
        f"Part (a), early advantage: **{d['part_a']}**; part (b), advantage gone late: **{d['part_b']}**.",
        "",
        "| h (steps) | h·dt / t_λ | strongest baseline | advantage [95% CI] |",
        "|---|---|---|---|",
    ]
    for r in d["rows"]:
        out.append(f"| {r['h']} | {r['x']:.3g} | {r['baseline']} | {_ci(r['advantage'])} |")
    if d["rows_actuated"]:
        out += [
            "",
            "Secondary (actuated test_in trajectories):",
            "",
            "| h | h·dt / t_λ | strongest baseline | advantage [95% CI] |",
            "|---|---|---|---|",
        ]
        for r in d["rows_actuated"]:
            out.append(f"| {r['h']} | {r['x']:.3g} | {r['baseline']} | {_ci(r['advantage'])} |")
    if d.get("drift_ratio_C_over_E") is not None:
        out += [
            "",
            f"Acrobot energy drift ratio C/E (descriptive, no threshold): {_ci(d['drift_ratio_C_over_E'])}.",
        ]
    return out


def _jsonable(v: Verdict) -> dict:
    def conv(x):
        if isinstance(x, CI):
            return x.as_dict()
        if isinstance(x, dict):
            return {k: conv(y) for k, y in x.items()}
        if isinstance(x, list):
            return [conv(y) for y in x]
        return x

    return {
        "hypothesis": v.hypothesis,
        "status": v.status,
        "rule": v.rule,
        "envs": [
            {"env": e.env, "status": e.status, "detail": conv(e.detail), "missing": e.missing} for e in v.envs
        ],
    }


def write_verdicts(results: str | Path, out: str | Path | None = None) -> list[Verdict]:
    results = Path(results)
    metrics = load_metrics(results)
    known = set(MODEL_IDS.values())
    exploratory = [
        Path(m["_path"]).parent.parent.name
        for m in metrics
        if int(m["seed"]) not in T.SEEDS or m["model"] not in known
    ]
    idx = index(m for m in metrics if int(m["seed"]) in T.SEEDS)
    lyap_path = results / f"lyapunov_{T.H4_ENV}.json"
    lyap = json.loads(lyap_path.read_text()) if lyap_path.exists() else None
    verdicts = [evaluate_h1(idx), evaluate_h2(idx), evaluate_h3(load_mbrl(results)), evaluate_h4(idx, lyap)]
    out = Path(out) if out else results / "verdicts.md"
    out.write_text(render(verdicts, exploratory))
    out.with_suffix(".json").write_text(json.dumps([_jsonable(v) for v in verdicts], indent=1, default=str))
    return verdicts
