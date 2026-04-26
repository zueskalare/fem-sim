"""Plugin: 2D binary image -> 2D FEM geometry tensor.

Converts a binary image (file or numpy array) into the standard
(5, H, W) geometry tensor used by the fem_sim pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fem_sim.geometry import _empty, _fill_material
from fem_sim.plugins import register_plugin

try:
    from PIL import Image as _PILImage
    _HAS_PILLOW = True
except ImportError:  # pragma: no cover
    _HAS_PILLOW = False


def _resize_nearest(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Nearest-neighbour resize for a 2-D array (pure numpy)."""
    h, w = arr.shape
    row_idx = np.linspace(0, h - 1, target_h).round().astype(int)
    col_idx = np.linspace(0, w - 1, target_w).round().astype(int)
    return arr[np.ix_(row_idx, col_idx)]


@register_plugin("image2d")
class Image2DPlugin:
    """Convert a 2D binary image into a 2D FEM geometry tensor."""

    name = "2D Binary Image"

    def validate(self, **kwargs: Any) -> list[str]:
        problems: list[str] = []
        has_path = "image_path" in kwargs
        has_array = "image_array" in kwargs

        if not has_path and not has_array:
            problems.append("Must provide either 'image_path' or 'image_array'")
        if has_path:
            p = Path(kwargs["image_path"])
            if not p.exists():
                problems.append(f"Image file not found: {p}")
            if not _HAS_PILLOW:
                problems.append("Pillow is required for file-based image loading "
                                "(pip install Pillow)")
        if has_array:
            arr = kwargs["image_array"]
            if not isinstance(arr, np.ndarray) or arr.ndim != 2:
                problems.append("'image_array' must be a 2-D numpy array")
        return problems

    def build(self, **kwargs: Any) -> np.ndarray:
        """Build geometry from a binary image.

        Parameters
        ----------
        image_path : str or Path, optional
            Path to PNG/JPG image file (requires Pillow).
        image_array : np.ndarray, optional
            2-D numpy array (H, W).  Takes precedence over *image_path*.
        threshold : float, default 0.5
            Values above this (after normalising to [0, 1]) are solid.
        invert : bool, default False
            If True, dark pixels become solid.
        target_h : int, optional
            Resize to this height (nearest-neighbour).
        target_w : int, optional
            Resize to this width (nearest-neighbour).
        E : float, default 210_000.0
        nu : float, default 0.3
        rho : float, default 7_800.0
        """
        # --- Load or accept image ---
        if "image_array" in kwargs:
            gray = np.asarray(kwargs["image_array"], dtype=np.float64)
        elif "image_path" in kwargs:
            if not _HAS_PILLOW:
                raise RuntimeError("Pillow is required for file-based image loading")
            img = _PILImage.open(kwargs["image_path"]).convert("L")
            gray = np.asarray(img, dtype=np.float64)
        else:
            raise ValueError("Must provide either 'image_path' or 'image_array'")

        # --- Normalise to [0, 1] ---
        vmin, vmax = gray.min(), gray.max()
        if vmax > vmin:
            gray = (gray - vmin) / (vmax - vmin)
        else:
            gray = np.ones_like(gray)

        # --- Optional resize ---
        target_h = kwargs.get("target_h")
        target_w = kwargs.get("target_w")
        if target_h is not None or target_w is not None:
            h, w = gray.shape
            th = target_h if target_h is not None else h
            tw = target_w if target_w is not None else w
            gray = _resize_nearest(gray, th, tw)

        # --- Threshold ---
        threshold = kwargs.get("threshold", 0.5)
        solid = gray > threshold

        # --- Optional invert ---
        if kwargs.get("invert", False):
            solid = ~solid

        # --- Build geometry tensor ---
        h, w = solid.shape
        E = kwargs.get("E", 210_000.0)
        nu = kwargs.get("nu", 0.3)
        rho = kwargs.get("rho", 7_800.0)

        geo = _empty(h, w)
        _fill_material(geo, solid, mat_id=1, E=E, nu=nu, rho=rho)
        return geo
