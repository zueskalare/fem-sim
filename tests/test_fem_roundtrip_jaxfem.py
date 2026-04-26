"""JAX-FEM roundtrip + physics tests.

Mirrors tests/test_fem_roundtrip.py but routes through ``backend='jaxfem'``.
Skips when jax-fem or petsc4py is not available.

Enable the full stack (macOS):
    brew install petsc
    uv sync --extra jaxfem
    uv pip install 'petsc4py==3.24.*'
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fem_sim.geometry import make_rectangle
from fem_sim.load_case import make_cantilever_distributed


def _jaxfem_stack_available() -> bool:
    for mod in ("jax_fem", "petsc4py"):
        if importlib.util.find_spec(mod) is None:
            return False
    try:
        importlib.import_module("jax_fem.solver")
    except Exception:
        return False
    return True


_SKIP = not _jaxfem_stack_available()
_REASON = "jax-fem / petsc4py not installed"


@unittest.skipIf(_SKIP, _REASON)
class TestJaxFemRoundTrip(unittest.TestCase):
    def test_rectangle_cantilever(self) -> None:
        from fem_sim.pixel_to_fem import run_simulation

        geo = make_rectangle(8, 16, E=210000, nu=0.3)
        bc = make_cantilever_distributed(geo, load_mag=-100)
        with tempfile.TemporaryDirectory() as tmp:
            sample = run_simulation(geo, bc, steps=3, run_dir=tmp, backend="jaxfem")

        self.assertEqual(sample.fields.shape, (3, 5, 8, 16))
        self.assertTrue(np.all(np.isfinite(sample.fields)))
        self.assertEqual(sample.metadata.get("backend"), "jaxfem")

        # Displacement should grow monotonically with the load ramp.
        disp_mag = np.sqrt(sample.fields[:, 0] ** 2 + sample.fields[:, 1] ** 2)
        max_disp = disp_mag.reshape(3, -1).max(axis=1)
        for i in range(1, 3):
            self.assertGreater(max_disp[i], max_disp[i - 1])

    def test_save_load_round_trip(self) -> None:
        from fem_sim.pixel_to_fem import load_sample, run_simulation, save_sample

        geo = make_rectangle(8, 16)
        bc = make_cantilever_distributed(geo, load_mag=-100)
        with tempfile.TemporaryDirectory() as tmp:
            sample = run_simulation(
                geo, bc, steps=2, run_dir=Path(tmp) / "run", backend="jaxfem"
            )
            npz = Path(tmp) / "sample.npz"
            save_sample(sample, npz)
            loaded = load_sample(npz)
            np.testing.assert_array_equal(loaded.geometry, sample.geometry)
            np.testing.assert_array_equal(loaded.boundary, sample.boundary)
            np.testing.assert_array_equal(loaded.fields, sample.fields)


@unittest.skipIf(_SKIP, _REASON)
class TestJaxFemCantileverPhysics(unittest.TestCase):
    """Mirror of the FreeFEM physics checks — same assertions, jaxfem backend."""

    def setUp(self) -> None:
        from fem_sim.pixel_to_fem import run_simulation
        self.run_simulation = run_simulation
        self.geo = make_rectangle(16, 32, E=210_000.0, nu=0.3, rho=7_800.0)

    def _solve(self, bc, steps=1, subdir="run"):
        def _inner(tmp):
            return self.run_simulation(
                self.geo, bc, steps=steps, run_dir=Path(tmp) / subdir, backend="jaxfem"
            )
        return _inner

    def test_max_deflection_at_free_end(self) -> None:
        bc = make_cantilever_distributed(self.geo, load_mag=-500, direction="y")
        with tempfile.TemporaryDirectory() as tmp:
            sample = self._solve(bc)(tmp)
            uy = sample.fields[0, 1]
            col = int(np.unravel_index(np.argmax(np.abs(uy)), uy.shape)[1])
            self.assertGreater(col, 16, f"max-deflection column = {col}")

    def test_fixed_end_near_zero_displacement(self) -> None:
        bc = make_cantilever_distributed(self.geo, load_mag=-500, direction="y")
        with tempfile.TemporaryDirectory() as tmp:
            sample = self._solve(bc)(tmp)
            # With penalty ~1e10 we expect residual on the order of load/penalty.
            np.testing.assert_allclose(sample.fields[0, 0, :, 0], 0.0, atol=1e-6)
            np.testing.assert_allclose(sample.fields[0, 1, :, 0], 0.0, atol=1e-6)

    def test_linearity(self) -> None:
        bc1 = make_cantilever_distributed(self.geo, load_mag=-250, direction="y")
        bc2 = make_cantilever_distributed(self.geo, load_mag=-500, direction="y")
        with tempfile.TemporaryDirectory() as tmp:
            s1 = self._solve(bc1, subdir="a")(tmp)
            s2 = self._solve(bc2, subdir="b")(tmp)
            r = float(np.max(np.abs(s2.fields[0, 1])) / np.max(np.abs(s1.fields[0, 1])))
            self.assertAlmostEqual(r, 2.0, places=1, msg=f"ratio={r:.3f}")

    def test_bending_stress_highest_near_fixed_end(self) -> None:
        bc = make_cantilever_distributed(self.geo, load_mag=-500, direction="y")
        with tempfile.TemporaryDirectory() as tmp:
            sample = self._solve(bc)(tmp)
            sxx = sample.fields[0, 2]
            w = sxx.shape[1]
            self.assertGreater(
                np.max(np.abs(sxx[:, : w // 4])),
                np.max(np.abs(sxx[:, 3 * w // 4 :])),
            )

    def test_bending_stress_sign_change_across_depth(self) -> None:
        bc = make_cantilever_distributed(self.geo, load_mag=-500, direction="y")
        with tempfile.TemporaryDirectory() as tmp:
            sample = self._solve(bc)(tmp)
            col = sample.fields.shape[-1] // 2
            sxx_col = sample.fields[0, 2, :, col]
            self.assertLess(
                sxx_col[0] * sxx_col[-1], 0,
                f"no sign change (bottom={sxx_col[0]:.3g}, top={sxx_col[-1]:.3g})",
            )

    def test_quasi_static_stepping_proportional(self) -> None:
        bc = make_cantilever_distributed(self.geo, load_mag=-500, direction="y")
        with tempfile.TemporaryDirectory() as tmp:
            sample = self._solve(bc, steps=5)(tmp)
            uy_all = sample.fields[:, 1]
            max_uy = np.array([np.max(np.abs(uy_all[i])) for i in range(5)])
            for i in range(1, 5):
                expected = (i + 1) / 1
                actual = float(max_uy[i] / max_uy[0])
                self.assertAlmostEqual(actual, expected, places=1,
                                       msg=f"step {i}: {actual:.2f} vs {expected:.1f}")


if __name__ == "__main__":
    unittest.main()
