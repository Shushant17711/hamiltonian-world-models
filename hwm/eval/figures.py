"""Every figure of the report, regenerated from ``results/`` (Req 10.3).

Static matplotlib figures for the README. Colour follows the model, never its rank: A-E keep the same
five categorical slots in every figure (palette validated for CVD separation; three slots are below 3:1
contrast on the light surface, so every series also carries a distinct marker and a direct label, and
the numbers are in ``verdicts.md`` as the table view). One y-axis per panel; recessive grid.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hwm.eval import thresholds as T
from hwm.eval.hypotheses import MODEL_IDS, index, load_mbrl, load_metrics

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100", "E": "#e87ba4"}
MARKERS = {"A": "o", "B": "s", "C": "^", "D": "D", "E": "P"}
LABELS = {"A": "A MLP", "B": "B PINN*", "C": "C latent ODE", "D": "D RSSM", "E": "E Hamiltonian"}
ENVS = ("pendulum", "cartpole", "acrobot", "orbit")


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "axes.titlesize": 10,
            "lines.linewidth": 2,
            "lines.markersize": 5,
            "legend.frameon": False,
        }
    )


def _seed_stats(vals: list[float]) -> tuple[float, float, float]:
    """Geometric mean and min/max over seeds (errors and drifts are compared on log axes)."""
    v = np.asarray([x for x in vals if np.isfinite(x) and x > 0])
    if not len(v):
        return math.nan, math.nan, math.nan
    return float(np.exp(np.log(v).mean())), float(v.min()), float(v.max())


def _label_end(ax, x, y, text, color) -> None:
    if np.isfinite(y):
        ax.annotate(
            text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7, color=INK2
        )
        ax.plot([], [], color=color)  # keeps the colour cycle untouched


def _runs(idx, env: str, letter: str) -> list[dict]:
    return [idx[k] for k in sorted(idx) if k[0] == env and k[1] == MODEL_IDS[letter] and k[2] == "state"]


def _mark_empty(fig) -> None:
    for ax in fig.axes:
        if not ax.lines and not ax.patches and not ax.collections:
            ax.text(0.5, 0.5, "no results yet", transform=ax.transAxes, ha="center", va="center", color=INK2)
            ax.set_yscale("linear")
            ax.set_xscale("linear")


def _save(fig, out: Path, name: str) -> Path:
    _mark_empty(fig)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- figures -----------------------------------------------------------------------------------------
def error_vs_horizon(idx, out: Path) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=False)
    for ax, env in zip(axes, ENVS, strict=True):
        for letter in "ABCDE":
            runs = _runs(idx, env, letter)
            curves = [r.get("nmse_curve", {}).get("test_long", {}) for r in runs]
            hs = sorted({int(h) for c in curves for h in c})
            if not hs:
                continue
            stats = [_seed_stats([c[str(h)] for c in curves if str(h) in c]) for h in hs]
            g, lo, hi = (np.array(x) for x in zip(*stats, strict=True))
            ax.fill_between(hs, lo, hi, color=COLORS[letter], alpha=0.15, linewidth=0)
            ax.plot(hs, g, color=COLORS[letter], marker=MARKERS[letter], label=LABELS[letter])
            _label_end(ax, hs[-1], g[-1], letter, COLORS[letter])
        ax.set(xscale="log", yscale="log", title=env, xlabel="rollout horizon h (steps)")
    axes[0].set_ylabel("nMSE (passive, train band)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.08, 1))
    fig.suptitle("Open-loop error vs horizon (geometric mean over seeds, band = min-max)", x=0.01, ha="left")
    return _save(fig, out, "error_vs_horizon")


def energy_drift(idx, out: Path) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, env in zip(axes, ENVS, strict=True):
        for letter in "ABCDE":
            curves = [r.get("drift_curve", {}).get("test_in") for r in _runs(idx, env, letter)]
            curves = [c for c in curves if c]
            if not curves:
                continue
            steps = curves[0]["steps"]
            g, lo, hi = (
                np.array(x)
                for x in zip(
                    *[_seed_stats([c["median"][j] for c in curves]) for j in range(len(steps))], strict=True
                )
            )
            ax.fill_between(steps, lo, hi, color=COLORS[letter], alpha=0.15, linewidth=0)
            ax.plot(steps, g, color=COLORS[letter], marker=MARKERS[letter], markevery=3, label=LABELS[letter])
            _label_end(ax, steps[-1], g[-1], letter, COLORS[letter])
        ax.set(xscale="log", yscale="log", title=env, xlabel="passive steps")
        ax.set_ylim(1e-5, 1e4)  # diverging seeds reach 1e40; clipped so the rest stays readable
    axes[0].set_ylabel("max |H(x_t) − H(x_0)| / E_ref")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.08, 1))
    fig.suptitle(
        "True-energy drift with u = 0 (median over 50 states; seeds: geo-mean, min-max; y clipped at 1e4; "
        "a seed drops out where its median diverges)",
        x=0.01,
        ha="left",
    )
    return _save(fig, out, "energy_drift")


def ood_bars(idx, out: Path) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    w = 0.38
    for ax, env in zip(axes, ENVS, strict=True):
        for i, letter in enumerate("ABCDE"):
            runs = _runs(idx, env, letter)
            for j, (split, alpha) in enumerate((("test_in", 0.45), ("test_ood", 1.0))):
                g, lo, hi = _seed_stats(
                    [r["nmse"][split]["100"] for r in runs if "100" in r["nmse"].get(split, {})]
                )
                if not np.isfinite(g):
                    continue
                x = i + (j - 0.5) * (w + 0.02)
                ax.bar(x, g, width=w, color=COLORS[letter], alpha=alpha)
                ax.errorbar(
                    x, g, yerr=[[max(g - lo, 0.0)], [max(hi - g, 0.0)]], color=INK2, linewidth=1, capsize=0
                )
        ax.set(yscale="log", title=env, xticks=range(5), xticklabels=list("ABCDE"))
    axes[0].set_ylabel("nMSE at h = 100")
    fig.suptitle(
        "Generalisation to unseen energies: in-band (light) vs OOD band (solid); B* uses the true dynamics",
        x=0.01,
        ha="left",
    )
    return _save(fig, out, "ood_bars")


def success_vs_steps(results: Path, out: Path) -> Path | None:
    runs = load_mbrl(results)
    if not runs:
        return None
    arms = {
        "hamiltonian_ens-pixels": ("E-ens (pixels)", COLORS["E"], "P"),
        "rssm-pixels": ("RSSM (pixels)", COLORS["D"], "D"),
        "hamiltonian_ens-state": ("E-ens (state)", COLORS["A"], "o"),
    }
    fig, axes = plt.subplots(1, len(T.H3_ENVS), figsize=(10, 3.2), sharey=True)
    for ax, env in zip(axes, T.H3_ENVS, strict=True):
        for key, (label, color, marker) in arms.items():
            recs = [v for k, v in runs.items() if k.startswith(f"mbrl-{env}-{key}-s")]
            if not recs:
                continue
            for r in recs:
                xs = [x["checkpoint"] for x in r]
                ax.plot(
                    xs,
                    [x["success_rate"] for x in r],
                    color=color,
                    marker=marker,
                    alpha=0.8,
                    label=label if r is recs[0] else None,
                )
        ax.axhline(T.H3_SUCCESS_THRESHOLD, color=INK2, linewidth=1, linestyle=":")
        ax.set(xscale="log", title=env, xlabel="env steps", ylim=(-0.05, 1.05))
    axes[0].set_ylabel("success rate (10 MPC episodes)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle("Sample efficiency of control (one line per seed; dotted: 80% threshold)", x=0.01, ha="left")
    return _save(fig, out, "success_vs_steps")


def reliability(metrics: list[dict], out: Path) -> Path | None:
    cal = [(m["env"], m["calibration"]["test_in"]) for m in metrics if m.get("calibration")]
    if not cal:
        return None
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    top = 0.0
    for i, (env, c) in enumerate(sorted(cal, key=lambda x: x[0])):
        r = c["reliability"]
        color = list(COLORS.values())[ENVS.index(env) if env in ENVS else i % 5]
        ax.plot(
            r["disagreement"],
            r["rmse"],
            color=color,
            marker="o",
            alpha=0.8,
            label=f"{env} (ρ = {c['spearman_pooled']:.2f})",
        )
        top = max(top, max(r["disagreement"]), max(r["rmse"]))
    ax.plot([0, top], [0, top], color=INK2, linewidth=1, linestyle=":")
    ax.set(
        xlabel="ensemble disagreement (bin mean)",
        ylabel="RMSE of the ensemble mean (bin)",
        title="Calibration of E-ens disagreement",
    )
    ax.legend(fontsize=7)
    return _save(fig, out, "reliability")


def advantage(results: Path, out: Path) -> Path | None:
    vj = results / "verdicts.json"
    if not vj.exists():
        return None
    h4 = next((v for v in json.loads(vj.read_text()) if v["hypothesis"] == "H4"), None)
    if not h4 or h4["status"] == "incomplete":
        return None
    d = h4["envs"][0]["detail"]
    rows = d["rows"]
    x = [r["x"] for r in rows]
    mean = [r["advantage"]["mean"] for r in rows]
    lo = [r["advantage"]["lo"] for r in rows]
    hi = [r["advantage"]["hi"] for r in rows]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.fill_between(x, lo, hi, color=COLORS["E"], alpha=0.2, linewidth=0)
    ax.plot(x, mean, color=COLORS["E"], marker="P", label="strongest baseline / E")
    ax.axhline(1, color=INK2, linewidth=1, linestyle=":")
    for v in (T.H4_EARLY_MAX_LYAPUNOV_TIMES, T.H4_LATE_MIN_LYAPUNOV_TIMES):
        ax.axvline(v, color=GRID, linewidth=1.5)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="horizon in Lyapunov times (h·dt / t_λ)",
        ylabel="advantage (95% CI)",
        title=f"Acrobot: H4 {h4['status']}",
    )
    return _save(fig, out, "advantage_vs_lyapunov")


def make_all(results: str | Path = "results", out: str | Path | None = None) -> list[Path]:
    _style()
    results = Path(results)
    out = Path(out) if out else results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(results)
    idx = index(m for m in metrics if int(m["seed"]) in T.SEEDS)
    made = [
        error_vs_horizon(idx, out),
        energy_drift(idx, out),
        ood_bars(idx, out),
        success_vs_steps(results, out),
        reliability(metrics, out),
        advantage(results, out),
    ]
    return [p for p in made if p is not None]
