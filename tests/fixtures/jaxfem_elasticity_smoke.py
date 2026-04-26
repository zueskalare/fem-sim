"""Smoke case for the jaxfem backend.

2D linear elasticity on a quad mesh:
  - Left edge fully clamped (Dirichlet u = 0)
  - Right edge pulled downward with a surface traction
  - Plane-strain, E = 70 GPa, nu = 0.3

Writes a VTU file to config.outputs['run_dir']/elasticity.vtu.

Invoked through JaxFemBackend with:

    SimulationConfig(
        case_script="tests/fixtures/jaxfem_elasticity_smoke.py",
        backend="jaxfem",
        outputs={"run_dir": "<out_dir>"},
    )
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from jax_fem.generate_mesh import Mesh, get_meshio_cell_type, rectangle_mesh
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol


class LinearElasticity(Problem):
    def get_tensor_map(self):
        def stress(u_grad):
            E = 70e3
            nu = 0.3
            mu = E / (2.0 * (1.0 + nu))
            lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            eps = 0.5 * (u_grad + u_grad.T)
            return lmbda * jnp.trace(eps) * jnp.eye(u_grad.shape[0]) + 2.0 * mu * eps
        return stress

    def get_surface_maps(self):
        def traction(_u, _x):
            return jnp.array([0.0, -100.0])
        return [traction]


def solve(config):
    ele_type = "QUAD4"
    cell_type = get_meshio_cell_type(ele_type)
    Lx, Ly = 1.0, 0.2
    Nx, Ny = 20, 5

    meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def left(point):
        return jnp.isclose(point[0], 0.0, atol=1e-5)

    def right(point):
        return jnp.isclose(point[0], Lx, atol=1e-5)

    def zero(_):
        return 0.0

    problem = LinearElasticity(
        mesh,
        vec=2,
        dim=2,
        ele_type=ele_type,
        dirichlet_bc_info=[[left, left], [0, 1], [zero, zero]],
        location_fns=[right],
    )

    sol_list = solver(problem)

    out_dir = Path(config.outputs.get("run_dir", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    vtk_path = out_dir / "elasticity.vtu"
    save_sol(problem.fes[0], sol_list[0], str(vtk_path))

    return {"vtk": str(vtk_path)}


if __name__ == "__main__":
    import argparse
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="outputs/jaxfem_smoke",
        help="Output directory for the VTU file (default: outputs/jaxfem_smoke)",
    )
    args = parser.parse_args()

    cfg = SimpleNamespace(outputs={"run_dir": args.run_dir})
    out = solve(cfg)
    print(f"wrote {out['vtk']}")
