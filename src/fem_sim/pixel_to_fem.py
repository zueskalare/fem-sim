"""Pixel-to-FEM bridge: convert pixel geometry + BCs to FreeFEM inputs,
run the solver, and read results back as (T, C_field, H, W) arrays.

Data flow:
  geometry (5, H, W) + boundary (6, H, W)
    -> write geometry.dat, boundary.dat as TSV
    -> invoke FreeFEM data/solvers/elasticity2d.edp
    -> read fields_step_*.tsv back into numpy arrays
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fem_sim.geometry import (
    CH_E,
    CH_NU,
    CH_RHO,
    CH_SOLID,
)
from fem_sim.load_case import (
    CH_DISP_MASK,
    CH_DX,
    CH_DY,
    CH_FORCE_MASK,
    CH_FX,
    CH_FY,
)
from fem_sim.freefem_binary import find_freefem_binary as _find_freefem_binary

# EDP script lives alongside this module in solvers/.
_EDP_SCRIPT = Path(__file__).resolve().parent / "solvers" / "elasticity2d.edp"

N_FIELD_CHANNELS = 5  # ux, uy, sxx, syy, sxy


@dataclass
class FEMSample:
    """A single dataset sample with aligned 2D tensors."""

    geometry: np.ndarray   # (C_geo, H, W)
    boundary: np.ndarray   # (C_bc, H, W)
    fields: np.ndarray     # (T, C_field, H, W)
    metadata: dict[str, Any]


def _pixel_grid_indices(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Row-major (iy outer, ix inner) flattened pixel index arrays."""
    iy, ix = np.mgrid[0:h, 0:w]
    return ix.ravel(), iy.ravel()


def write_geometry_dat(geo: np.ndarray, path: Path) -> None:
    """Write geometry array to TSV data file for FreeFEM.

    Format: ix iy solid E nu rho  (one row per pixel, row-major order).
    FreeFEM parses with ``ifstream >> ix >> ...``, so any whitespace works.
    """
    _, h, w = geo.shape
    ix, iy = _pixel_grid_indices(h, w)
    rows = np.column_stack([
        ix, iy,
        geo[CH_SOLID].ravel(),
        geo[CH_E].ravel(),
        geo[CH_NU].ravel(),
        geo[CH_RHO].ravel(),
    ])
    np.savetxt(path, rows, fmt=["%d", "%d", "%.6g", "%.6g", "%.6g", "%.6g"], delimiter="\t")


def write_boundary_dat(bc: np.ndarray, path: Path) -> None:
    """Write BC array to TSV data file for FreeFEM.

    Format: ix iy disp_mask force_mask dx dy fx fy
    """
    _, h, w = bc.shape
    ix, iy = _pixel_grid_indices(h, w)
    rows = np.column_stack([
        ix, iy,
        bc[CH_DISP_MASK].ravel(),
        bc[CH_FORCE_MASK].ravel(),
        bc[CH_DX].ravel(),
        bc[CH_DY].ravel(),
        bc[CH_FX].ravel(),
        bc[CH_FY].ravel(),
    ])
    np.savetxt(
        path, rows,
        fmt=["%d", "%d", "%.6g", "%.6g", "%.6g", "%.6g", "%.6g", "%.6g"],
        delimiter="\t",
    )


def load_field_tsv(path: Path, h: int, w: int) -> np.ndarray:
    """Read a fields_step_*.tsv file into a (C_field, H, W) array.

    Expected columns: ix iy ux uy sxx syy sxy (with header row).
    """
    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2 + N_FIELD_CHANNELS:
        raise ValueError(
            f"{path}: expected ≥ {2 + N_FIELD_CHANNELS} columns, "
            f"got {data.shape[1]}"
        )
    ix = data[:, 0].astype(int)
    iy = data[:, 1].astype(int)
    in_bounds = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    ix, iy = ix[in_bounds], iy[in_bounds]
    field = np.zeros((N_FIELD_CHANNELS, h, w), dtype=np.float64)
    for ch in range(N_FIELD_CHANNELS):
        field[ch, iy, ix] = data[in_bounds, 2 + ch]
    return field


def load_results(run_dir: Path, h: int, w: int, steps: int) -> np.ndarray:
    """Read all field TSV files from a run into (T, C_field, H, W)."""
    fields = []
    for step in range(steps):
        path = run_dir / f"fields_step_{step}.tsv"
        if not path.exists():
            break
        fields.append(load_field_tsv(path, h, w))
    if not fields:
        raise FileNotFoundError(f"No field files found in {run_dir}")
    return np.stack(fields, axis=0)


def run_simulation(
    geo: np.ndarray,
    bc: np.ndarray,
    steps: int = 10,
    run_dir: Path | str | None = None,
    edp_script: Path | str | None = None,
    freefem_binary: str | None = None,
    backend: str = "freefem",
) -> FEMSample:
    """Full pipeline: write data files, run the FEM solver, read results.

    Parameters
    ----------
    geo : (C_geo, H, W) geometry array.
    bc : (C_bc, H, W) boundary condition array.
    steps : int
        Number of quasi-static load increments.
    run_dir : Path, optional
        Output directory. Created if needed. Defaults to a temp dir.
    edp_script : Path, optional
        Override path to the EDP script (``freefem`` backend only).
    freefem_binary : str, optional
        Override FreeFEM binary name/path (``freefem`` backend only).
    backend : {"freefem", "jaxfem"}
        Solver backend.  "jaxfem" requires ``jax-fem`` + ``petsc4py``.
        Both backends produce identical ``fields_step_N.tsv`` output.

    Returns
    -------
    FEMSample with geometry, boundary, fields, and metadata.
    """
    _, h, w = geo.shape

    if run_dir is None:
        import tempfile
        run_dir = Path(tempfile.mkdtemp(prefix="fem_pixel_"))
    else:
        run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write data files — same format for both backends.
    write_geometry_dat(geo, run_dir / "geometry.dat")
    write_boundary_dat(bc, run_dir / "boundary.dat")

    if backend == "freefem":
        _run_freefem(w, h, steps, run_dir, edp_script, freefem_binary)
    elif backend == "jaxfem":
        _run_jaxfem(w, h, steps, run_dir)
    else:
        raise ValueError(
            f"unknown backend {backend!r}; expected 'freefem' or 'jaxfem'"
        )

    fields = load_results(run_dir, h, w, steps)

    metadata = {
        "nx": w,
        "ny": h,
        "steps": steps,
        "run_dir": str(run_dir),
        "backend": backend,
    }

    # Save metadata.
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return FEMSample(
        geometry=geo,
        boundary=bc,
        fields=fields,
        metadata=metadata,
    )


def _run_freefem(
    w: int,
    h: int,
    steps: int,
    run_dir: Path,
    edp_script: Path | str | None,
    freefem_binary: str | None,
) -> None:
    edp = Path(edp_script) if edp_script else _EDP_SCRIPT
    binary = _find_freefem_binary(freefem_binary)
    cmd = [
        binary,
        "-nw",
        str(edp),
        "-nx", str(w),
        "-ny", str(h),
        "-steps", str(steps),
        "-run_dir", str(run_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(run_dir))
    if result.returncode != 0:
        raise RuntimeError(
            f"FreeFEM failed (rc={result.returncode}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr: {result.stderr[:2000]}"
        )


def _run_jaxfem(w: int, h: int, steps: int, run_dir: Path) -> None:
    try:
        from fem_sim.solvers.elasticity2d_jaxfem import solve as jaxfem_solve
    except ImportError as exc:
        raise ImportError(
            "jaxfem backend requires jax-fem and petsc4py; "
            "install with `uv sync --extra jaxfem` plus system PETSc"
        ) from exc
    jaxfem_solve(nx=w, ny=h, steps=steps, run_dir=run_dir)


def save_sample(sample: FEMSample, path: Path | str) -> Path:
    """Save an FEMSample as a .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        geometry=sample.geometry,
        boundary=sample.boundary,
        fields=sample.fields,
    )
    # Save metadata alongside.
    meta_path = path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(sample.metadata, f, indent=2)
    return path


def load_sample(path: Path | str) -> FEMSample:
    """Load an FEMSample from a .npz file."""
    path = Path(path)
    data = np.load(path)
    meta_path = path.with_suffix(".json")
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    return FEMSample(
        geometry=data["geometry"],
        boundary=data["boundary"],
        fields=data["fields"],
        metadata=metadata,
    )


