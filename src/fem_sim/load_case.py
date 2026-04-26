"""Boundary condition / load case generators for 2D FEM dataset.

Each generator takes a geometry array (C_geo, H, W) and returns a
boundary condition tensor (C_bc, H, W) where:
  Ch 0: disp_mask  — 1.0 where displacement BC is applied
  Ch 1: force_mask — 1.0 where force BC is applied
  Ch 2: dx         — prescribed displacement in x
  Ch 3: dy         — prescribed displacement in y
  Ch 4: fx         — applied force in x
  Ch 5: fy         — applied force in y

Convention: the geometry pixel grid uses row 0 = bottom of domain.
  Left edge:   column 0
  Right edge:  column W-1
  Bottom edge: row 0
  Top edge:    row H-1
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fem_sim.geometry import CH_SOLID

N_BC_CHANNELS = 6
CH_DISP_MASK = 0
CH_FORCE_MASK = 1
CH_DX = 2
CH_DY = 3
CH_FX = 4
CH_FY = 5


def _empty_bc(h: int, w: int) -> np.ndarray:
    return np.zeros((N_BC_CHANNELS, h, w), dtype=np.float64)


_EDGE_SLICES = {
    "left":   (slice(None), 0),
    "right":  (slice(None), -1),
    "bottom": (0,           slice(None)),
    "top":    (-1,          slice(None)),
}


def _solid_edge(geo: np.ndarray, edge: str) -> np.ndarray:
    """Boolean (H, W) mask of solid pixels along the named edge.

    edge ∈ {'left', 'right', 'bottom', 'top'}.
    """
    if edge not in _EDGE_SLICES:
        raise ValueError(f"unknown edge {edge!r}; expected {list(_EDGE_SLICES)}")
    h, w = geo.shape[1], geo.shape[2]
    mask = np.zeros((h, w), dtype=bool)
    sel = _EDGE_SLICES[edge]
    mask[sel] = geo[CH_SOLID][sel] > 0.5
    return mask


def _apply_distributed_load(
    bc: np.ndarray,
    edge_solid_1d: np.ndarray,
    edge_slice: tuple,
    load_mag: float,
    direction: str,
) -> None:
    """Spread *load_mag* evenly across the solid pixels on an edge.

    edge_slice is the same kind of (row_sel, col_sel) tuple used in _EDGE_SLICES.
    edge_solid_1d is a 1-D boolean mask along the edge (H- or W-shaped).
    """
    n_loaded = int(edge_solid_1d.sum())
    if n_loaded == 0:
        return
    per_pixel = load_mag / n_loaded
    bc[CH_FORCE_MASK][edge_slice][edge_solid_1d] = 1.0
    ch = CH_FY if direction == "y" else CH_FX
    bc[ch][edge_slice][edge_solid_1d] = per_pixel


# ------------------------------------------------------------------
# Load case generators
# ------------------------------------------------------------------


def make_cantilever_point_load(
    geo: np.ndarray,
    load_mag: float = -1000.0,
    load_pos: float = 1.0,
    direction: str = "y",
) -> np.ndarray:
    """Cantilever beam: left edge fixed, point load on right edge.

    Parameters
    ----------
    geo : (C_geo, H, W)
        Geometry array.
    load_mag : float
        Load magnitude (negative = downward for direction='y').
    load_pos : float
        Normalised position along the loaded edge [0, 1].
        0.0 = bottom of right edge, 1.0 = top of right edge.
    direction : 'x' or 'y'
        Load direction.

    Returns
    -------
    np.ndarray, shape (6, H, W)
    """
    h, w = geo.shape[1], geo.shape[2]
    bc = _empty_bc(h, w)

    # Fixed left edge (displacement BC: ux=0, uy=0).
    bc[CH_DISP_MASK][_solid_edge(geo, "left")] = 1.0

    # Point load on the chosen pixel of the right edge.
    right_solid = np.where(geo[CH_SOLID, :, -1] > 0.5)[0]
    if len(right_solid) > 0:
        idx = right_solid[int(load_pos * (len(right_solid) - 1))]
        bc[CH_FORCE_MASK, idx, -1] = 1.0
        ch = CH_FY if direction == "y" else CH_FX
        bc[ch, idx, -1] = load_mag

    return bc


def make_cantilever_distributed(
    geo: np.ndarray,
    load_mag: float = -500.0,
    direction: str = "y",
) -> np.ndarray:
    """Cantilever beam: left edge fixed, distributed load on right edge."""
    h, w = geo.shape[1], geo.shape[2]
    bc = _empty_bc(h, w)
    bc[CH_DISP_MASK][_solid_edge(geo, "left")] = 1.0
    _apply_distributed_load(
        bc,
        edge_solid_1d=geo[CH_SOLID, :, -1] > 0.5,
        edge_slice=_EDGE_SLICES["right"],
        load_mag=load_mag,
        direction=direction,
    )
    return bc


def make_simply_supported_distributed(
    geo: np.ndarray,
    load_mag: float = -800.0,
    direction: str = "y",
) -> np.ndarray:
    """Pinned at both bottom corners, distributed load on the top edge.

    Note: both supports are pinned in x and y (over-constrained vs a true
    simply supported beam, which would leave one support free in x).  Kept
    pinned for solver stability.
    """
    h, w = geo.shape[1], geo.shape[2]
    bc = _empty_bc(h, w)

    if geo[CH_SOLID, 0, 0] > 0.5:
        bc[CH_DISP_MASK, 0, 0] = 1.0
    if geo[CH_SOLID, 0, -1] > 0.5:
        bc[CH_DISP_MASK, 0, -1] = 1.0

    _apply_distributed_load(
        bc,
        edge_solid_1d=geo[CH_SOLID, -1, :] > 0.5,
        edge_slice=_EDGE_SLICES["top"],
        load_mag=load_mag,
        direction=direction,
    )
    return bc


def make_displacement_bc(
    geo: np.ndarray,
    disp_mag: float = 0.01,
    direction: str = "y",
) -> np.ndarray:
    """Cantilever with prescribed displacement on the right edge."""
    h, w = geo.shape[1], geo.shape[2]
    bc = _empty_bc(h, w)

    bc[CH_DISP_MASK][_solid_edge(geo, "left")] = 1.0

    right = _solid_edge(geo, "right")
    bc[CH_DISP_MASK][right] = 1.0
    ch = CH_DY if direction == "y" else CH_DX
    bc[ch][right] = disp_mag
    return bc


def make_top_load_fixed_bottom(
    geo: np.ndarray,
    load_mag: float = -1000.0,
    direction: str = "y",
) -> np.ndarray:
    """Fixed bottom edge, distributed load on the top edge."""
    h, w = geo.shape[1], geo.shape[2]
    bc = _empty_bc(h, w)
    bc[CH_DISP_MASK][_solid_edge(geo, "bottom")] = 1.0
    _apply_distributed_load(
        bc,
        edge_solid_1d=geo[CH_SOLID, -1, :] > 0.5,
        edge_slice=_EDGE_SLICES["top"],
        load_mag=load_mag,
        direction=direction,
    )
    return bc


# ------------------------------------------------------------------
# Characterization tests (uniaxial / shear / biaxial)
#
# These prescribe displacement on opposing edges so the central region
# experiences a near-uniform stress state.  Because the BC tensor uses a
# single per-pixel `disp_mask` (no per-component mask), every clamped edge
# constrains BOTH ux and uy — corners are over-constrained and cannot
# Poisson-contract laterally.  This matches the existing load cases in
# this module and is the standard FEM textbook setup; the central region
# remains a clean characterization signal for ML training.
# ------------------------------------------------------------------


def make_uniaxial(
    geo: np.ndarray,
    disp_mag: float = 0.01,
    direction: str = "x",
) -> np.ndarray:
    """Uniaxial tension or compression on a square (or rectangular) sample.

    Sign convention (engineering): ``disp_mag > 0`` → tension,
    ``disp_mag < 0`` → compression.

    Parameters
    ----------
    geo : (C_geo, H, W)
    disp_mag : float
        Prescribed displacement magnitude on the loaded edge.
    direction : 'x' or 'y'
        Axis of the prescribed displacement.

        - ``'x'``: left edge clamped (``ux=uy=0``); right edge clamped at
          ``(dx=disp_mag, dy=0)``.  Top/bottom traction-free.
        - ``'y'``: bottom edge clamped; top edge clamped at
          ``(dx=0, dy=disp_mag)``.  Left/right traction-free.

    Returns
    -------
    np.ndarray, shape (6, H, W)
    """
    h, w = geo.shape[1], geo.shape[2]
    bc = _empty_bc(h, w)

    if direction == "x":
        bc[CH_DISP_MASK][_solid_edge(geo, "left")] = 1.0
        right = _solid_edge(geo, "right")
        bc[CH_DISP_MASK][right] = 1.0
        bc[CH_DX][right] = disp_mag
    elif direction == "y":
        bc[CH_DISP_MASK][_solid_edge(geo, "bottom")] = 1.0
        top = _solid_edge(geo, "top")
        bc[CH_DISP_MASK][top] = 1.0
        bc[CH_DY][top] = disp_mag
    else:
        raise ValueError(f"direction must be 'x' or 'y', got {direction!r}")

    return bc


def make_shear(
    geo: np.ndarray,
    disp_mag: float = 0.01,
    direction: str = "x",
) -> np.ndarray:
    """Simple shear test on a square (or rectangular) sample.

    The loaded edge slides parallel to itself relative to the clamped
    opposite edge — no thickness change at the loaded edge.  Sign of
    ``disp_mag`` flips the shear direction.

    Parameters
    ----------
    geo : (C_geo, H, W)
    disp_mag : float
        Prescribed sliding displacement on the loaded edge.
    direction : 'x' or 'y'
        Axis of the sliding displacement.

        - ``'x'`` (horizontal shear): bottom edge clamped (``ux=uy=0``);
          top edge clamped at ``(dx=disp_mag, dy=0)``.
        - ``'y'`` (vertical shear): left edge clamped; right edge clamped
          at ``(dx=0, dy=disp_mag)``.

    Returns
    -------
    np.ndarray, shape (6, H, W)
    """
    h, w = geo.shape[1], geo.shape[2]
    bc = _empty_bc(h, w)

    if direction == "x":
        bc[CH_DISP_MASK][_solid_edge(geo, "bottom")] = 1.0
        top = _solid_edge(geo, "top")
        bc[CH_DISP_MASK][top] = 1.0
        bc[CH_DX][top] = disp_mag
    elif direction == "y":
        bc[CH_DISP_MASK][_solid_edge(geo, "left")] = 1.0
        right = _solid_edge(geo, "right")
        bc[CH_DISP_MASK][right] = 1.0
        bc[CH_DY][right] = disp_mag
    else:
        raise ValueError(f"direction must be 'x' or 'y', got {direction!r}")

    return bc


def make_biaxial(
    geo: np.ndarray,
    disp_x: float = 0.01,
    disp_y: float = 0.01,
) -> np.ndarray:
    """Biaxial tension / compression test on a square (or rectangular) sample.

    Layered edge writes give the natural biaxial corner displacements:

    - bottom-left  ``(0, 0)``        — pinned (both bottom + left set it to 0)
    - bottom-right ``(disp_x, 0)``   — right edge overwrites dx; dy stays 0
    - top-left     ``(0, disp_y)``   — top edge overwrites dy; dx stays 0
    - top-right    ``(disp_x, disp_y)`` — top sets dy, then right sets dx

    For equi-biaxial tension pass ``disp_x == disp_y > 0``.  For
    equi-biaxial compression use negative values.  Mixed tension-compression
    (e.g. ``disp_x > 0, disp_y < 0``) is supported.

    Parameters
    ----------
    geo : (C_geo, H, W)
    disp_x : float
        Prescribed x-displacement on the right edge.
    disp_y : float
        Prescribed y-displacement on the top edge.

    Returns
    -------
    np.ndarray, shape (6, H, W)
    """
    h, w = geo.shape[1], geo.shape[2]
    bc = _empty_bc(h, w)

    # Pin the two reference edges (dx = dy = 0 by default).
    bc[CH_DISP_MASK][_solid_edge(geo, "bottom")] = 1.0
    bc[CH_DISP_MASK][_solid_edge(geo, "left")] = 1.0

    # Top edge: stretch in y.  Don't touch dx — for the top-left pixel, the
    # left edge already set dx=0; for the top-right pixel, the right edge
    # below will set dx=disp_x.
    top = _solid_edge(geo, "top")
    bc[CH_DISP_MASK][top] = 1.0
    bc[CH_DY][top] = disp_y

    # Right edge: stretch in x.  Same logic in reverse: dy at top-right
    # remains disp_y from the top-edge write above.
    right = _solid_edge(geo, "right")
    bc[CH_DISP_MASK][right] = 1.0
    bc[CH_DX][right] = disp_x

    return bc


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------


def save_load_case(bc: np.ndarray, path: str | Path) -> Path:
    """Save BC array to .npy."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, bc)
    return p


def load_load_case(path: str | Path) -> np.ndarray:
    """Load BC array from .npy."""
    return np.load(path)
