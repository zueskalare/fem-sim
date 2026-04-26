"""Geometry pixel image generators for 2D FEM dataset.

Each generator returns a (C_geo, H, W) numpy array where:
  Ch 0: solid_mask  — 1.0 where material exists, 0.0 for void
  Ch 1: material_id — integer ID (0=void, 1=mat_A, 2=mat_B, ...)
  Ch 2: E           — Young's modulus at each pixel
  Ch 3: nu          — Poisson's ratio at each pixel
  Ch 4: rho         — density at each pixel

Coordinates: pixel (i, j) maps to physical position
  x = j * (lx / W),  y = i * (ly / H)
with origin at bottom-left.  Row 0 is the bottom of the domain.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

N_GEO_CHANNELS = 5
CH_SOLID = 0
CH_MATID = 1
CH_E = 2
CH_NU = 3
CH_RHO = 4


def _empty(h: int, w: int) -> np.ndarray:
    """Return a zeroed (C_geo, H, W) array."""
    return np.zeros((N_GEO_CHANNELS, h, w), dtype=np.float64)


def _fill_material(
    geo: np.ndarray,
    mask: np.ndarray,
    mat_id: int,
    E: float,
    nu: float,
    rho: float,
) -> None:
    """Fill material properties into *geo* wherever *mask* is True."""
    geo[CH_SOLID][mask] = 1.0
    geo[CH_MATID][mask] = float(mat_id)
    geo[CH_E][mask] = E
    geo[CH_NU][mask] = nu
    geo[CH_RHO][mask] = rho


# ------------------------------------------------------------------
# Generators
# some simple shapes for testing and dataset generation; more can be added as needed
# ------------------------------------------------------------------


def make_rectangle(
    h: int,
    w: int,
    E: float = 210_000.0,
    nu: float = 0.3,
    rho: float = 7_800.0,
) -> np.ndarray:
    """Solid rectangle — all pixels are material.

    Parameters
    ----------
    h, w : int
        Grid height and width in pixels.
    E, nu, rho : float
        Material properties applied uniformly.

    Returns
    -------
    np.ndarray, shape (5, h, w)
    """
    geo = _empty(h, w)
    mask = np.ones((h, w), dtype=bool)
    _fill_material(geo, mask, mat_id=1, E=E, nu=nu, rho=rho)
    return geo


def make_plate_with_hole(
    h: int,
    w: int,
    cx: float,
    cy: float,
    r: float,
    E: float = 210_000.0,
    nu: float = 0.3,
    rho: float = 7_800.0,
) -> np.ndarray:
    """Rectangular plate with a circular hole.

    Parameters
    ----------
    h, w : int
        Grid size in pixels.
    cx, cy : float
        Hole centre in normalised coordinates [0, 1].
    r : float
        Hole radius in normalised coordinates (fraction of min(h, w)).
    E, nu, rho : float
        Material properties.

    Returns
    -------
    np.ndarray, shape (5, h, w)
    """
    geo = _empty(h, w)
    # Pixel centre coordinates normalised to [0, 1].
    yy, xx = np.mgrid[0:h, 0:w]
    xn = (xx + 0.5) / w
    yn = (yy + 0.5) / h
    r_abs = r * min(h, w) / max(h, w)  # keep circular in normalised space
    dist = np.sqrt((xn - cx) ** 2 + (yn - cy) ** 2)
    solid = dist > r_abs
    _fill_material(geo, solid, mat_id=1, E=E, nu=nu, rho=rho)
    return geo


def make_lshape(
    h: int,
    w: int,
    cut_frac: float = 0.5,
    E: float = 210_000.0,
    nu: float = 0.3,
    rho: float = 7_800.0,
) -> np.ndarray:
    """L-shaped domain — full rectangle minus a rectangular cutout.

    The cutout removes the top-right quadrant (or fraction thereof).

    Parameters
    ----------
    h, w : int
        Grid size.
    cut_frac : float
        Fraction of each dimension removed from top-right corner (0, 1).
    E, nu, rho : float
        Material properties.

    Returns
    -------
    np.ndarray, shape (5, h, w)
    """
    geo = _empty(h, w)
    solid = np.ones((h, w), dtype=bool)
    cut_h = int(h * cut_frac)
    cut_w = int(w * cut_frac)
    # Remove top-right block.  Row 0 = bottom, so top rows are h-cut_h..h.
    if cut_h > 0 and cut_w > 0:
        solid[(h - cut_h) :, (w - cut_w) :] = False
    _fill_material(geo, solid, mat_id=1, E=E, nu=nu, rho=rho)
    return geo


def make_porous(
    h: int,
    w: int,
    n_pores: int,
    pore_r_range: tuple[float, float] = (0.03, 0.08),
    seed: int = 42,
    E: float = 210_000.0,
    nu: float = 0.3,
    rho: float = 7_800.0,
    margin: float = 0.1,
) -> np.ndarray:
    """Rectangular plate with randomly placed circular pores.

    Parameters
    ----------
    h, w : int
        Grid size.
    n_pores : int
        Number of pores to place.
    pore_r_range : (float, float)
        Min and max pore radius in normalised coordinates.
    seed : int
        Random seed for reproducibility.
    margin : float
        Pore centres are kept at least *margin* from the boundary (normalised).
    E, nu, rho : float
        Material properties.

    Returns
    -------
    np.ndarray, shape (5, h, w)
    """
    geo = _empty(h, w)
    rng = np.random.default_rng(seed)

    yy, xx = np.mgrid[0:h, 0:w]
    xn = (xx + 0.5) / w
    yn = (yy + 0.5) / h

    solid = np.ones((h, w), dtype=bool)
    r_min, r_max = pore_r_range
    for _ in range(n_pores):
        px = rng.uniform(margin, 1.0 - margin)
        py = rng.uniform(margin, 1.0 - margin)
        pr = rng.uniform(r_min, r_max)
        dist = np.sqrt((xn - px) ** 2 + (yn - py) ** 2)
        solid[dist <= pr] = False

    _fill_material(geo, solid, mat_id=1, E=E, nu=nu, rho=rho)
    return geo


def make_bimat_rectangle(
    h: int,
    w: int,
    split_frac: float = 0.5,
    E1: float = 210_000.0,
    nu1: float = 0.3,
    rho1: float = 7_800.0,
    E2: float = 70_000.0,
    nu2: float = 0.33,
    rho2: float = 2_700.0,
) -> np.ndarray:
    """Bi-material rectangle — two materials separated by a vertical line.

    Parameters
    ----------
    h, w : int
        Grid size.
    split_frac : float
        Fraction of width occupied by material 1 (left side).
    E1, nu1, rho1 : float
        Left material properties.
    E2, nu2, rho2 : float
        Right material properties.

    Returns
    -------
    np.ndarray, shape (5, h, w)
    """
    geo = _empty(h, w)
    split_col = int(w * split_frac)

    mask1 = np.zeros((h, w), dtype=bool)
    mask1[:, :split_col] = True
    _fill_material(geo, mask1, mat_id=1, E=E1, nu=nu1, rho=rho1)

    mask2 = np.zeros((h, w), dtype=bool)
    mask2[:, split_col:] = True
    _fill_material(geo, mask2, mat_id=2, E=E2, nu=nu2, rho=rho2)

    return geo


def make_grf_bimat(
    h: int,
    w: int,
    correlation_length: float = 4.0,
    volume_fraction: float = 0.5,
    seed: int = 42,
    E_A: float = 30.0,
    nu_A: float = 0.48,
    rho_A: float = 1.2e-9,
    E_B: float = 3500.0,
    nu_B: float = 0.36,
    rho_B: float = 1.24e-9,
) -> np.ndarray:
    """Two-phase pattern from a thresholded Gaussian Random Field.

    Defaults model a TPU (phase A, soft) / PLA (phase B, stiff) composite
    in MPa–mm–ms units (E in MPa, density in Mg/mm^3).

    Parameters
    ----------
    h, w : int
        Grid size in pixels.
    correlation_length : float
        Feature size in pixels.  The GRF is low-pass filtered with a
        Gaussian whose width in pixel units is this value.  Larger →
        smoother/blobbier patterns; smaller → fine-grained speckle.
    volume_fraction : float
        Target fraction of phase A (mat_id=1).  The field is thresholded
        at its empirical quantile so the resulting mask has exactly
        (modulo tie-breaking) ``volume_fraction`` phase-A pixels.
    seed : int
        RNG seed for reproducibility.
    E_A, nu_A, rho_A : float
        Phase-A material (TPU defaults).
    E_B, nu_B, rho_B : float
        Phase-B material (PLA defaults).

    Returns
    -------
    np.ndarray, shape (5, h, w).  Fully solid (both phases are material).
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w))

    # Spectral Gaussian filter.  Frequencies are in cycles/pixel; a spatial
    # Gaussian of width σ (pixels) ↔ multiplying the spectrum by
    # exp(-0.5 (2π σ k)^2).
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]
    k2 = kx ** 2 + ky ** 2
    kernel = np.exp(-0.5 * (2.0 * np.pi * correlation_length) ** 2 * k2)

    field = np.real(np.fft.ifft2(np.fft.fft2(noise) * kernel))
    # Standardise so the threshold via quantile is well behaved.
    field = (field - field.mean()) / (field.std() + 1e-12)

    threshold = np.quantile(field, volume_fraction)
    mask_A = field <= threshold

    geo = _empty(h, w)
    _fill_material(geo, mask_A, mat_id=1, E=E_A, nu=nu_A, rho=rho_A)
    _fill_material(geo, ~mask_A, mat_id=2, E=E_B, nu=nu_B, rho=rho_B)
    return geo


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------


def save_geometry(geo: np.ndarray, path: str | Path) -> Path:
    """Save geometry array to a .npy file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, geo)
    return path


def load_geometry(path: str | Path) -> np.ndarray:
    """Load geometry array from a .npy file."""
    return np.load(path)
