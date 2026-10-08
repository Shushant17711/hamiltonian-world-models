"""Model registry: ``build(cfg) -> WorldModel`` (design §4).

``cfg`` is a full run config with at least ``env`` (env name) and ``model.name``. Models register
themselves with ``@register("name")`` and are constructed as ``cls(cfg.model, env, obs_mode)``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from hwm.envs import make as make_env
from hwm.models.base import WorldModel

_REGISTRY: dict[str, type[WorldModel]] = {}

# model name -> module that defines it; imported lazily so a missing model does not break the rest
_MODULES = {
    "mlp": "hwm.models.mlp",
    "pinn": "hwm.models.pinn",
    "latent_ode": "hwm.models.latent_ode",
    "rssm": "hwm.models.rssm",
    "hamiltonian": "hwm.models.hamiltonian",
    "ensemble": "hwm.models.ensemble",
}


def register(name: str) -> Callable[[type[WorldModel]], type[WorldModel]]:
    def deco(cls: type[WorldModel]) -> type[WorldModel]:
        _REGISTRY[name] = cls
        return cls

    return deco


def get(name: str) -> type[WorldModel]:
    if name not in _REGISTRY and name in _MODULES:
        importlib.import_module(_MODULES[name])
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {sorted(set(_REGISTRY) | set(_MODULES))}")
    return _REGISTRY[name]


def build(cfg: Any) -> WorldModel:
    """Construct the model named by ``cfg.model.name`` for env ``cfg.env``."""
    env = make_env(cfg["env"])
    obs_mode = cfg.get("obs_mode", "state")
    return get(cfg["model"]["name"])(cfg["model"], env, obs_mode)
