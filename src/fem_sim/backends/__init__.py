from __future__ import annotations

from fem_sim.backends.base import BackendRunner

_REGISTRY: dict[str, type[BackendRunner]] = {}


def register(name: str):
    """Decorator that registers a backend class under the given name."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_backend(name: str) -> BackendRunner:
    """Return an instance of the named backend.

    Raises ValueError for unknown backend names.
    Imports are deferred so optional deps (dolfinx) don't break the package.
    """
    if name not in _REGISTRY:
        # Lazy-load built-in backends on first request
        _load_builtins()
    if name not in _REGISTRY:
        raise ValueError(f"unknown backend {name!r}; available: {list_backends()}")
    return _REGISTRY[name]()


def list_backends() -> list[str]:
    """Return names of all registered backends."""
    _load_builtins()
    return sorted(_REGISTRY)


def _load_builtins() -> None:
    if "freefem" not in _REGISTRY:
        from fem_sim.backends import freefem as _  # noqa: F401
    if "fenicsx" not in _REGISTRY:
        try:
            from fem_sim.backends import fenicsx as _  # noqa: F401
        except ImportError:
            pass  # dolfinx not installed — skip silently
    if "jaxfem" not in _REGISTRY:
        try:
            from fem_sim.backends import jaxfem as _  # noqa: F401
        except ImportError:
            pass  # jax-fem not installed — skip silently
