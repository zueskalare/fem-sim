"""JAX-FEM implementation of the 2D plane-stress elasticity solver.

Mirrors the I/O contract of ``solvers/elasticity2d.edp``:

- Reads ``geometry.dat`` and ``boundary.dat`` from ``run_dir``
- Solves plane-stress linear elasticity with quasi-static load ramping
- Writes ``fields_step_N.tsv``, ``series.tsv``, ``summary.tsv`` in the
  same column format so ``pixel_to_fem.load_results`` can consume them.

Invariants that ``pixel_to_fem.py`` relies on:

- ``fields_step_N.tsv`` columns: ``ix iy ux uy sxx syy sxy`` (header row).
- One row per pixel, row-major ``(iy, ix)``.
- File is created for each ``istep`` in ``range(steps)``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as onp

logger = logging.getLogger(__name__)

from jax_fem.generate_mesh import Mesh, get_meshio_cell_type, rectangle_mesh
from jax_fem.problem import Problem
from jax_fem.solver import solver as _jaxfem_solver

_ELE_TYPE = "QUAD4"
_PENALTY = 1e10
_VOID_E = 1e-6


class PixelElasticity(Problem):
    """Plane-stress linear elasticity with per-cell material + BC params.

    ``internal_vars`` layout:
      [0] mat : (nc, nq, 2) — (E, nu) per quad point
      [1] bc  : (nc, nq, 5) — (fx, fy, dmask, dx, dy) per quad point
    """

    def get_tensor_map(self):
        def stress(u_grad, mat, _bc):
            E, nu = mat[0], mat[1]
            mu = E / (2.0 * (1.0 + nu))
            lam2d = 2.0 * mu * nu / (1.0 - nu + 1e-30)
            eps = 0.5 * (u_grad + u_grad.T)
            return lam2d * jnp.trace(eps) * jnp.eye(2) + 2.0 * mu * eps
        return stress

    def get_mass_map(self):
        def mass(u, _x, _mat, bc):
            fx, fy, dmask, dx, dy = bc[0], bc[1], bc[2], bc[3], bc[4]
            body = jnp.array([-fx, -fy])
            dirichlet = dmask * _PENALTY * jnp.array([u[0] - dx, u[1] - dy])
            return body + dirichlet
        return mass

    def set_params(self, params):
        mat, bc = params
        self.internal_vars = [mat, bc]


def _read_columns(path: Path, ncols: int) -> onp.ndarray:
    """Read a whitespace-separated file into a (N, ncols) array."""
    data = onp.loadtxt(str(path))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < ncols:
        raise ValueError(f"{path}: expected >={ncols} columns, got {data.shape[1]}")
    return data


def _scatter_columns(data: onp.ndarray, nx: int, ny: int, cols: range) -> list[onp.ndarray]:
    """Scatter rows of (N, ≥max(cols)+1) into (ny, nx) arrays per requested column."""
    ix = data[:, 0].astype(int)
    iy = data[:, 1].astype(int)
    out: list[onp.ndarray] = []
    for col in cols:
        arr = onp.zeros((ny, nx))
        arr[iy, ix] = data[:, col]
        out.append(arr)
    return out


def _read_geometry(run_dir: Path, nx: int, ny: int):
    data = _read_columns(run_dir / "geometry.dat", 6)
    solid, E, nu = _scatter_columns(data, nx, ny, range(2, 5))
    return solid, E, nu


def _read_boundary(run_dir: Path, nx: int, ny: int):
    data = _read_columns(run_dir / "boundary.dat", 8)
    dmask, fmask, dx, dy, fx, fy = _scatter_columns(data, nx, ny, range(2, 8))
    return dmask, fmask, dx, dy, fx, fy


def _pixel_to_cells(arr: onp.ndarray) -> onp.ndarray:
    """(ny, nx) pixel array -> (nx*ny,) in rectangle_mesh cell order (ix*ny + iy)."""
    return arr.T.ravel()


def _broadcast_quads(cell_vals: jnp.ndarray, num_quads: int) -> jnp.ndarray:
    """(nc, C) -> (nc, nq, C)."""
    return jnp.broadcast_to(cell_vals[:, None, :], (cell_vals.shape[0], num_quads, cell_vals.shape[1]))


def solve(nx: int, ny: int, steps: int, run_dir: Path | str) -> None:
    """Run the pixel-grid elasticity solve.

    Reads ``geometry.dat`` / ``boundary.dat`` from ``run_dir`` and writes
    ``fields_step_N.tsv``, ``series.tsv``, ``summary.tsv`` back to the
    same directory.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    _, E_pix, nu_pix = _read_geometry(run_dir, nx, ny)
    dmask_pix, fmask_pix, dx_pix, dy_pix, fx_pix, fy_pix = _read_boundary(run_dir, nx, ny)

    # Void pixels (E == 0) would make the stiffness matrix singular.
    # Use an ersatz material to keep the system solvable.
    E_pix = onp.where(E_pix > 0, E_pix, _VOID_E)
    # Plane-stress lambda → ∞ as nu → 0.5 (incompressible). Cap just below
    # 0.5 to keep the assembly finite, but warn if user data needed clamping
    # so silent rubber-modelling errors are visible.
    nu_max = 0.499
    nu_min = 0.0
    out_of_range = (nu_pix < nu_min) | (nu_pix > nu_max)
    if out_of_range.any():
        logger.warning(
            "elasticity2d_jaxfem: %d pixels had nu outside (%.3f, %.3f]; "
            "clamping. Use a different formulation for incompressible solids.",
            int(out_of_range.sum()), nu_min, nu_max,
        )
    nu_pix = onp.clip(nu_pix, nu_min, nu_max)

    Lx = 1.0
    Ly = ny / nx
    meshio_mesh = rectangle_mesh(Nx=nx, Ny=ny, domain_x=Lx, domain_y=Ly)
    cell_type = get_meshio_cell_type(_ELE_TYPE)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    problem = PixelElasticity(mesh, vec=2, dim=2, ele_type=_ELE_TYPE)
    num_cells = problem.fes[0].num_cells
    num_quads = problem.fes[0].num_quads

    E_cells = jnp.asarray(_pixel_to_cells(E_pix))
    nu_cells = jnp.asarray(_pixel_to_cells(nu_pix))
    mat_cells = jnp.stack([E_cells, nu_cells], axis=-1)
    mat_quad = _broadcast_quads(mat_cells, num_quads)

    fx_eff = _pixel_to_cells(fx_pix * fmask_pix)
    fy_eff = _pixel_to_cells(fy_pix * fmask_pix)
    dmask_cells = _pixel_to_cells(dmask_pix)
    dx_cells = _pixel_to_cells(dx_pix)
    dy_cells = _pixel_to_cells(dy_pix)

    cells_np = onp.array(problem.fes[0].cells)
    series = ["step\tload_fraction\tmax_ux\tmax_uy"]

    for istep in range(steps):
        alpha = (istep + 1.0) / steps
        bc_cells = jnp.asarray(onp.stack([
            alpha * fx_eff,
            alpha * fy_eff,
            dmask_cells,
            alpha * dx_cells,
            alpha * dy_cells,
        ], axis=-1))
        bc_quad = _broadcast_quads(bc_cells, num_quads)

        problem.set_params((mat_quad, bc_quad))
        # UMFPACK (direct sparse LU) is more robust than the default
        # bicgstab iterative solver when E contrast is high (TPU vs PLA)
        # or the load is localized (point loads); matrix size is small.
        sol = _jaxfem_solver(problem, solver_options={"umfpack_solver": {}})[0]

        ux_pix, uy_pix = _sample_displacement(sol, cells_np, nx, ny)
        sxx_pix, syy_pix, sxy_pix = _sample_stress(problem, sol, E_cells, nu_cells, nx, ny)

        _write_fields_tsv(
            run_dir / f"fields_step_{istep}.tsv",
            nx, ny,
            ux_pix, uy_pix, sxx_pix, syy_pix, sxy_pix,
        )
        series.append(
            f"{istep}\t{alpha:.6g}\t"
            f"{float(onp.max(onp.abs(ux_pix))):.6g}\t"
            f"{float(onp.max(onp.abs(uy_pix))):.6g}"
        )

    (run_dir / "series.tsv").write_text("\n".join(series) + "\n")
    (run_dir / "summary.tsv").write_text(
        f"nx\t{nx}\nny\t{ny}\nsteps\t{steps}\nlx\t{Lx}\nly\t{Ly}\n"
    )


def _cells_to_pixels(cell_vals: onp.ndarray, nx: int, ny: int) -> onp.ndarray:
    """Inverse of _pixel_to_cells: (nx*ny,) cell-order -> (ny, nx) pixel grid."""
    return cell_vals.reshape(nx, ny).T


def _sample_displacement(sol, cells_np, nx, ny):
    """QUAD4 center = mean of 4 corner nodes (bilinear shape functions evaluate
    to 1/4 each at ξ=η=0)."""
    sol_np = onp.asarray(sol)                          # (n_nodes, 2)
    cell_disp = sol_np[cells_np].mean(axis=1)          # (n_cells, 2)
    ux = _cells_to_pixels(cell_disp[:, 0], nx, ny)
    uy = _cells_to_pixels(cell_disp[:, 1], nx, ny)
    return ux, uy


def _sample_stress(problem, sol, E_cells, nu_cells, nx, ny):
    """Cell-center stress. For bilinear QUAD4, mean of 2×2 Gauss-point gradients
    equals the centroid gradient."""
    u_grads = problem.fes[0].sol_to_grad(sol)          # (nc, nq, 2, 2)
    u_grad_cells = jnp.mean(u_grads, axis=1)           # (nc, 2, 2)

    def stress_at(u_grad, E, nu):
        mu = E / (2.0 * (1.0 + nu))
        lam2d = 2.0 * mu * nu / (1.0 - nu + 1e-30)
        eps = 0.5 * (u_grad + u_grad.T)
        return lam2d * jnp.trace(eps) * jnp.eye(2) + 2.0 * mu * eps

    sigma = jax.vmap(stress_at)(u_grad_cells, E_cells, nu_cells)  # (nc, 2, 2)
    sigma_np = onp.asarray(sigma)
    sxx = _cells_to_pixels(sigma_np[:, 0, 0], nx, ny)
    syy = _cells_to_pixels(sigma_np[:, 1, 1], nx, ny)
    sxy = _cells_to_pixels(sigma_np[:, 0, 1], nx, ny)
    return sxx, syy, sxy


def _write_fields_tsv(path, nx, ny, ux, uy, sxx, syy, sxy) -> None:
    iy_grid, ix_grid = onp.mgrid[0:ny, 0:nx]
    rows = onp.column_stack([
        ix_grid.ravel(), iy_grid.ravel(),
        ux.ravel(), uy.ravel(),
        sxx.ravel(), syy.ravel(), sxy.ravel(),
    ])
    with open(path, "w") as f:
        f.write("ix\tiy\tux\tuy\tsxx\tsyy\tsxy\n")
        onp.savetxt(
            f, rows,
            fmt=["%d", "%d", "%.6g", "%.6g", "%.6g", "%.6g", "%.6g"],
            delimiter="\t",
        )
