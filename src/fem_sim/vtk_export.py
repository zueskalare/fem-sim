"""Export an FEMSample as a time-series VTK collection.

Writes one ``.vti`` (VTK ImageData) per load step plus a ``.pvd`` collection
that ParaView opens as an animation.  ImageData is a perfect fit for the
pixel grid — regular spacing, no mesh / cell connectivity to declare — so
the writer is hand-rolled XML with no extra dependencies.

Per-frame channels (CellData on the H × W × 1 voxel grid):

* Static (constant across steps): ``solid_mask``, ``material_id``, ``E``,
  ``nu``, ``rho``, ``disp_mask``, ``force_mask``, ``prescribed_disp``
  (vector), ``prescribed_force`` (vector).
* Time-varying: ``displacement`` (vector), ``stress_xx``, ``stress_yy``,
  ``stress_xy``.  At step ``t`` these come from ``sample.fields[t]``.

Open the generated ``.pvd`` in ParaView to see the load ramp animate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from fem_sim.geometry import CH_E, CH_MATID, CH_NU, CH_RHO, CH_SOLID
from fem_sim.load_case import (
    CH_DISP_MASK,
    CH_DX,
    CH_DY,
    CH_FORCE_MASK,
    CH_FX,
    CH_FY,
)
from fem_sim.pixel_to_fem import FEMSample, load_sample


def export_sample_vtk(
    sample: str | Path | FEMSample,
    output_dir: str | Path | None = None,
    name: str | None = None,
) -> Path:
    """Export a sample as a time-series of ``.vti`` files plus a ``.pvd``.

    Parameters
    ----------
    sample
        Either the path to a ``.npz`` sample (loaded via :func:`load_sample`)
        or an in-memory :class:`FEMSample`.
    output_dir
        Directory to write into.  Created if missing.  If ``None`` and
        ``sample`` is a path, defaults to ``<sample_dir>/<stem>_vtk/`` next
        to the ``.npz``.  Required when ``sample`` is an in-memory object.
    name
        Stem for the ``.pvd`` file.  Defaults to the sample's path stem
        when one is available, otherwise ``"sample"``.

    Returns
    -------
    Path
        The ``.pvd`` collection file.  Open it in ParaView to play the
        animation.
    """
    if isinstance(sample, (str, Path)):
        sample_path = Path(sample)
        loaded = load_sample(sample_path)
        if output_dir is None:
            output_dir = sample_path.parent / f"{sample_path.stem}_vtk"
        if name is None:
            name = sample_path.stem
    else:
        loaded = sample
        if output_dir is None:
            raise ValueError("output_dir is required when sample is an in-memory FEMSample")
        if name is None:
            name = "sample"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, h, w = loaded.geometry.shape
    n_steps = loaded.fields.shape[0]

    static = _static_arrays(loaded)
    step_paths: list[Path] = []
    for t in range(n_steps):
        data = dict(static)
        ux = loaded.fields[t, 0]
        uy = loaded.fields[t, 1]
        data["displacement"] = (ux, uy)
        data["stress_xx"] = loaded.fields[t, 2]
        data["stress_yy"] = loaded.fields[t, 3]
        data["stress_xy"] = loaded.fields[t, 4]
        step_path = output_dir / f"step_{t:04d}.vti"
        _write_vti(step_path, h, w, data)
        step_paths.append(step_path)

    pvd_path = output_dir / f"{name}.pvd"
    # Use load fraction as the timestep value: step 0 → 1/T, ..., step T-1 → 1.
    timesteps = [(t + 1) / n_steps for t in range(n_steps)]
    _write_pvd(pvd_path, step_paths, timesteps)
    return pvd_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _static_arrays(sample: FEMSample) -> dict[str, np.ndarray | tuple[np.ndarray, ...]]:
    """Per-pixel arrays that don't change with load step."""
    geo = sample.geometry
    bc = sample.boundary
    return {
        "solid_mask":       geo[CH_SOLID],
        "material_id":      geo[CH_MATID],
        "E":                geo[CH_E],
        "nu":               geo[CH_NU],
        "rho":              geo[CH_RHO],
        "disp_mask":        bc[CH_DISP_MASK],
        "force_mask":       bc[CH_FORCE_MASK],
        "prescribed_disp":  (bc[CH_DX], bc[CH_DY]),
        "prescribed_force": (bc[CH_FX], bc[CH_FY]),
    }


def _write_vti(
    path: Path,
    h: int,
    w: int,
    cell_data: Mapping[str, np.ndarray | tuple[np.ndarray, ...]],
) -> None:
    """Write a single .vti (VTK ImageData) file for an H × W pixel grid.

    cell_data values are either:
      - a (H, W) array — written as a scalar CellData array
      - a tuple of (H, W) arrays — written as a vector CellData array
        (ParaView treats len-2 tuples as 2D vectors with z=0 implicit).
    """
    arrays_xml = "\n".join(_format_data_array(name, arr) for name, arr in cell_data.items())
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n'
        f'  <ImageData WholeExtent="0 {w} 0 {h} 0 0" Origin="0 0 0" Spacing="1 1 1">\n'
        f'    <Piece Extent="0 {w} 0 {h} 0 0">\n'
        '      <CellData>\n'
        f'{arrays_xml}\n'
        '      </CellData>\n'
        '    </Piece>\n'
        '  </ImageData>\n'
        '</VTKFile>\n',
        encoding="utf-8",
    )


def _format_data_array(name: str, arr: np.ndarray | tuple[np.ndarray, ...]) -> str:
    """Render one CellData <DataArray> element.

    Cell ordering for VTK ImageData of extent (0,W,0,H,0,0) is
    ix-fastest, then iy, which matches numpy C-order ravel of (iy, ix).
    """
    if isinstance(arr, tuple):
        ncomp = len(arr)
        # interleave per pixel: (a0_x, a0_y, a1_x, a1_y, ...)
        flat = np.stack(arr, axis=-1).ravel()
        comp_attr = f' NumberOfComponents="{ncomp}"'
    else:
        flat = arr.ravel()
        comp_attr = ""
    values = " ".join(f"{v:.6g}" for v in flat)
    return (
        f'        <DataArray type="Float64" Name="{name}"{comp_attr} format="ascii">\n'
        f'          {values}\n'
        f'        </DataArray>'
    )


def _write_pvd(
    path: Path,
    step_paths: list[Path],
    timesteps: list[float],
) -> None:
    """Write a ParaView .pvd collection referencing each step's .vti.

    Uses paths relative to the .pvd file so the bundle is portable.
    """
    base = path.parent
    entries = "\n".join(
        f'    <DataSet timestep="{ts:.6g}" file="{step.relative_to(base).as_posix()}"/>'
        for ts, step in zip(timesteps, step_paths)
    )
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
        '  <Collection>\n'
        f'{entries}\n'
        '  </Collection>\n'
        '</VTKFile>\n',
        encoding="utf-8",
    )
