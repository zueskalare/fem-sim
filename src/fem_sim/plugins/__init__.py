"""Geometry plugin registry.

Mirrors the lazy-load + decorator pattern from ``fem_sim.backends``.
"""

from __future__ import annotations

from fem_sim.plugins.base import GeoPlugin

_REGISTRY: dict[str, type[GeoPlugin]] = {}


def register_plugin(name: str):
    """Decorator that registers a geometry plugin class under *name*."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_plugin(name: str) -> GeoPlugin:
    """Return an instance of the named plugin.

    Raises ``ValueError`` for unknown plugin names.
    """
    if name not in _REGISTRY:
        _load_builtins()
    if name not in _REGISTRY:
        raise ValueError(f"unknown plugin {name!r}; available: {list_plugins()}")
    return _REGISTRY[name]()


def list_plugins() -> list[str]:
    """Return sorted names of all registered plugins."""
    _load_builtins()
    return sorted(_REGISTRY)


def _load_builtins() -> None:
    if "image2d" not in _REGISTRY:
        try:
            from fem_sim.plugins import image2d as _  # noqa: F401
        except ImportError:
            pass  # Pillow not installed — skip silently
