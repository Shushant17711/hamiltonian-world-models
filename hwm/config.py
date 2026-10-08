"""YAML configs as nested, frozen, attribute-accessible mappings.

Configs stay schema-light on purpose: each component reads the keys it needs and supplies its own
defaults, so a new model does not require editing a central dataclass. What the rest of the code
relies on is:

* ``load_config(path, overrides)`` - read YAML, merge ``extends`` parents, apply ``a.b=c`` overrides
* ``Config`` - read-only, attribute access (``cfg.model.hidden``), ``.get(key, default)``
* ``config_hash(cfg)`` - stable sha1 of the resolved content (key order does not matter)
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads YAML-1.2 floats such as ``3e-4`` (PyYAML would return a string)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.VERBOSE,
    ),
    list("-+0123456789."),
)


def _yaml_load(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)


class Config(Mapping[str, Any]):
    """Immutable nested mapping with attribute access."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any] | None = None):
        frozen = {k: _freeze(v) for k, v in (data or {}).items()}
        object.__setattr__(self, "_data", frozen)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"config has no key {key!r} (keys: {sorted(self._data)})") from None

    def __setattr__(self, key: str, value: Any) -> None:
        raise TypeError("Config is immutable; use with_overrides()")

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Config({self.to_dict()!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return self.to_dict() == _thaw(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(config_hash(self))

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self)

    def with_overrides(self, overrides: Mapping[str, Any] | list[str] | None) -> Config:
        data = self.to_dict()
        if overrides is None:
            return Config(data)
        items = [_parse_override(o) for o in overrides] if isinstance(overrides, list) else overrides.items()
        for dotted, value in items:
            _set_dotted(data, dotted, value)
        return Config(data)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=True))


def _freeze(v: Any) -> Any:
    if isinstance(v, Config):
        return v
    if isinstance(v, Mapping):
        return Config(v)
    if isinstance(v, list | tuple):
        return tuple(_freeze(x) for x in v)
    return v


def _thaw(v: Any) -> Any:
    if isinstance(v, Mapping):
        return {k: _thaw(x) for k, x in v.items()}
    if isinstance(v, tuple | list):
        return [_thaw(x) for x in v]
    return v


def _parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"override must look like a.b=value, got {text!r}")
    key, raw = text.split("=", 1)
    return key.strip(), _yaml_load(raw)


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = data
    for p in parts[:-1]:
        nxt = node.setdefault(p, {})
        if not isinstance(nxt, dict):
            raise TypeError(f"cannot set {dotted!r}: {p!r} is not a mapping")
        node = nxt
    node[parts[-1]] = _thaw(value)


def _deep_merge(base: dict[str, Any], top: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in top.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = _thaw(v)
    return out


def _load_yaml_with_extends(path: Path) -> dict[str, Any]:
    raw = _yaml_load(path.read_text()) or {}
    parents = raw.pop("extends", None)
    if parents is None:
        return raw
    if isinstance(parents, str):
        parents = [parents]
    merged: dict[str, Any] = {}
    for parent in parents:
        merged = _deep_merge(merged, _load_yaml_with_extends((path.parent / parent).resolve()))
    return _deep_merge(merged, raw)


def load_config(path: str | Path, overrides: list[str] | Mapping[str, Any] | None = None) -> Config:
    """Load a YAML config; ``extends: [other.yaml]`` merges parents (relative paths) first."""
    return Config(_load_yaml_with_extends(Path(path))).with_overrides(overrides)


def config_hash(cfg: Mapping[str, Any], length: int = 12) -> str:
    blob = json.dumps(_thaw(cfg), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:length]
