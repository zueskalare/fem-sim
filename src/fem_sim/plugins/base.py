"""Base protocol for geometry plugins."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class GeoPlugin(Protocol):
    """Protocol every geometry plugin must satisfy.

    A plugin converts some external input (image, CAD file, etc.)
    into the standard (5, H, W) geometry tensor consumed by the
    rest of the fem_sim pipeline.
    """

    name: str

    def build(self, **kwargs: Any) -> np.ndarray:
        """Build a geometry tensor from the given parameters.

        Returns
        -------
        np.ndarray, shape (5, H, W)
            Channels: solid_mask, material_id, E, nu, rho.
        """
        ...

    def validate(self, **kwargs: Any) -> list[str]:
        """Return a list of parameter problems.  Empty list means valid."""
        ...
