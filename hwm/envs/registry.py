from hwm.envs.base import Env

_REGISTRY: dict[str, type[Env]] = {}


def register(cls: type[Env]) -> type[Env]:
    _REGISTRY[cls.name] = cls
    return cls


def make(name: str, **kwargs) -> Env:
    _load_builtin()
    if name not in _REGISTRY:
        raise KeyError(f"unknown env {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def names() -> list[str]:
    _load_builtin()
    return sorted(_REGISTRY)


def _load_builtin() -> None:
    if _REGISTRY:
        return
    from hwm.envs.orbit import Orbit
    from hwm.envs.pendulum import Pendulum

    for cls in (Pendulum, Orbit):
        register(cls)
