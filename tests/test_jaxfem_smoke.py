"""End-to-end smoke test for the jaxfem backend.

Runs the elasticity fixture through the backend and checks that a VTU
file is produced.  Skips cleanly when jax-fem or its PETSc dependency
is not installed.

Enable on macOS:

    brew install petsc
    uv sync --extra jaxfem
    uv pip install petsc4py
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from fem_sim import get_backend
from fem_sim.config import SimulationConfig


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "jaxfem_elasticity_smoke.py"


def _jaxfem_stack_available() -> bool:
    for mod in ("jax_fem", "petsc4py"):
        if importlib.util.find_spec(mod) is None:
            return False
    # Confirm the solver module is actually importable (not just jax_fem).
    try:
        importlib.import_module("jax_fem.solver")
    except Exception:
        return False
    return True


@unittest.skipUnless(_jaxfem_stack_available(), "jax-fem / petsc4py not installed")
class TestJaxFemSmoke(unittest.TestCase):
    def test_elasticity_produces_vtk(self) -> None:
        self.assertTrue(FIXTURE.exists(), f"fixture missing: {FIXTURE}")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            cfg = SimulationConfig(
                case_script=str(FIXTURE),
                backend="jaxfem",
                outputs={"run_dir": str(run_dir)},
            )
            result = get_backend("jaxfem").run(cfg)

            self.assertTrue(result.succeeded,
                            f"run failed (rc={result.return_code})")
            self.assertEqual(result.backend, "jaxfem")
            self.assertIn("vtk", result.outputs, f"outputs={result.outputs}")

            vtk_path = Path(result.outputs["vtk"])
            self.assertTrue(vtk_path.exists(), f"VTK not written: {vtk_path}")
            self.assertGreater(vtk_path.stat().st_size, 0,
                               "VTK file is empty")


if __name__ == "__main__":
    unittest.main()
