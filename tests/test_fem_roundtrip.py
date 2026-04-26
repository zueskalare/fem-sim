"""FreeFEM round-trip and physics validation tests.

These tests require FreeFem++ and are auto-skipped when absent.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fem_sim.geometry import make_rectangle
from fem_sim.load_case import make_cantilever_point_load, make_cantilever_distributed


def _freefem_available() -> bool:
    """Check if FreeFem++ is available."""
    for name in ("FreeFem++", "freefem++"):
        if shutil.which(name):
            return True
    # macOS app bundle (various version layouts).
    for p in Path("/Applications/FreeFem++.app/Contents").glob("**/FreeFem++"):
        if p.is_file():
            return True
    return False


@unittest.skipUnless(_freefem_available(), "FreeFem++ not installed")
class TestFEMRoundTrip(unittest.TestCase):
    """Test the full pixel -> FEM -> pixel pipeline."""

    def test_rectangle_cantilever(self):
        """Basic shape and finiteness check."""
        from fem_sim.pixel_to_fem import run_simulation

        geo = make_rectangle(8, 16, E=210000, nu=0.3)
        bc = make_cantilever_point_load(geo, load_mag=-100, load_pos=1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            sample = run_simulation(geo, bc, steps=3, run_dir=tmpdir)

            self.assertEqual(sample.fields.shape, (3, 5, 8, 16))
            self.assertTrue(np.all(np.isfinite(sample.fields)))

            # Displacement should increase with load step (linear ramp).
            disp_mag = np.sqrt(sample.fields[:, 0] ** 2 + sample.fields[:, 1] ** 2)
            max_disp = disp_mag.reshape(3, -1).max(axis=1)
            for i in range(1, 3):
                self.assertGreater(max_disp[i], max_disp[i - 1])

    def test_save_load_round_trip(self):
        """Verify .npz serialisation preserves data exactly."""
        from fem_sim.pixel_to_fem import (
            run_simulation,
            save_sample,
            load_sample,
        )

        geo = make_rectangle(8, 16)
        bc = make_cantilever_distributed(geo, load_mag=-100)

        with tempfile.TemporaryDirectory() as tmpdir:
            sample = run_simulation(geo, bc, steps=2, run_dir=Path(tmpdir) / "run")
            npz_path = Path(tmpdir) / "test_sample.npz"
            save_sample(sample, npz_path)

            loaded = load_sample(npz_path)
            np.testing.assert_array_equal(loaded.geometry, sample.geometry)
            np.testing.assert_array_equal(loaded.boundary, sample.boundary)
            np.testing.assert_array_equal(loaded.fields, sample.fields)


@unittest.skipUnless(_freefem_available(), "FreeFem++ not installed")
class TestCantileverBeamPhysics(unittest.TestCase):
    """Validate FEM results against known cantilever beam behaviour.

    A cantilever beam (fixed left, loaded right) of steel under a tip
    load should produce:
      - Maximum deflection at the free end (right side)
      - Fixed end has zero displacement
      - Deflection proportional to load (linearity)
      - Bending stress (sxx) highest near the fixed end
      - sxx changes sign across the beam depth (tension on one side,
        compression on the other)
    """

    def setUp(self):
        from fem_sim.pixel_to_fem import run_simulation

        self.run_simulation = run_simulation

        # Steel cantilever, 32x16 pixels (aspect ratio 2:1).
        self.geo = make_rectangle(16, 32, E=210_000.0, nu=0.3, rho=7_800.0)

    def test_max_deflection_at_free_end(self):
        """Maximum vertical displacement should be at the free (right) end."""
        bc = make_cantilever_distributed(self.geo, load_mag=-500, direction="y")

        with tempfile.TemporaryDirectory() as tmpdir:
            sample = self.run_simulation(self.geo, bc, steps=1, run_dir=tmpdir)

            uy = sample.fields[0, 1]  # (H, W) vertical displacement at full load
            # Column index of max |uy| should be on the right half.
            max_pos = np.unravel_index(np.argmax(np.abs(uy)), uy.shape)
            col = max_pos[1]
            self.assertGreater(col, 16, f"Max deflection at col {col}, expected > 16")

    def test_fixed_end_near_zero_displacement(self):
        """Displacement at the fixed (left) edge should be near zero."""
        bc = make_cantilever_distributed(self.geo, load_mag=-500, direction="y")

        with tempfile.TemporaryDirectory() as tmpdir:
            sample = self.run_simulation(self.geo, bc, steps=1, run_dir=tmpdir)

            ux_left = sample.fields[0, 0, :, 0]   # ux at left column
            uy_left = sample.fields[0, 1, :, 0]   # uy at left column
            # Should be close to zero (penalty BC, not exact zero).
            np.testing.assert_allclose(ux_left, 0.0, atol=1e-8,
                                       err_msg="Left edge ux should be ~0")
            np.testing.assert_allclose(uy_left, 0.0, atol=1e-8,
                                       err_msg="Left edge uy should be ~0")

    def test_linearity_double_load_double_displacement(self):
        """Doubling the load should double the displacement (linear elasticity)."""
        bc1 = make_cantilever_distributed(self.geo, load_mag=-250, direction="y")
        bc2 = make_cantilever_distributed(self.geo, load_mag=-500, direction="y")

        with tempfile.TemporaryDirectory() as tmpdir:
            s1 = self.run_simulation(self.geo, bc1, steps=1,
                                     run_dir=Path(tmpdir) / "load1")
            s2 = self.run_simulation(self.geo, bc2, steps=1,
                                     run_dir=Path(tmpdir) / "load2")

            # Fields at full load (step 0, since steps=1).
            uy1 = s1.fields[0, 1]  # vertical disp, load=250
            uy2 = s2.fields[0, 1]  # vertical disp, load=500

            max1 = np.max(np.abs(uy1))
            max2 = np.max(np.abs(uy2))
            ratio = max2 / max1
            self.assertAlmostEqual(ratio, 2.0, places=1,
                                   msg=f"Displacement ratio {ratio:.3f}, expected 2.0")

    def test_bending_stress_highest_near_fixed_end(self):
        """sxx magnitude should be largest near the fixed support."""
        bc = make_cantilever_distributed(self.geo, load_mag=-500, direction="y")

        with tempfile.TemporaryDirectory() as tmpdir:
            sample = self.run_simulation(self.geo, bc, steps=1, run_dir=tmpdir)

            sxx = sample.fields[0, 2]  # (H, W)
            # Compare max |sxx| in the left quarter vs the right quarter.
            w = sxx.shape[1]
            left_max = np.max(np.abs(sxx[:, :w // 4]))
            right_max = np.max(np.abs(sxx[:, 3 * w // 4:]))
            self.assertGreater(left_max, right_max,
                               "Bending stress should be higher near fixed end")

    def test_bending_stress_sign_change_across_depth(self):
        """sxx should be positive on one side of the neutral axis and
        negative on the other (bending)."""
        bc = make_cantilever_distributed(self.geo, load_mag=-500, direction="y")

        with tempfile.TemporaryDirectory() as tmpdir:
            sample = self.run_simulation(self.geo, bc, steps=1, run_dir=tmpdir)

            sxx = sample.fields[0, 2]  # (H, W)
            # Check at a column in the middle of the beam.
            mid_col = sxx.shape[1] // 2
            sxx_col = sxx[:, mid_col]
            # Top and bottom should have opposite signs.
            self.assertLess(sxx_col[0] * sxx_col[-1], 0,
                            "sxx should change sign across beam depth "
                            f"(bottom={sxx_col[0]:.4g}, top={sxx_col[-1]:.4g})")

    def test_quasi_static_stepping_proportional(self):
        """Fields at each load step should scale proportionally."""
        bc = make_cantilever_distributed(self.geo, load_mag=-500, direction="y")

        with tempfile.TemporaryDirectory() as tmpdir:
            sample = self.run_simulation(self.geo, bc, steps=5, run_dir=tmpdir)

            # Step i has load_fraction = (i+1)/5.
            # Displacement should be proportional to load_fraction.
            uy_all = sample.fields[:, 1]  # (5, H, W)
            max_uy = np.array([np.max(np.abs(uy_all[i])) for i in range(5)])
            # Ratios step_i / step_0 should be (i+1)/1.
            for i in range(1, 5):
                expected_ratio = (i + 1) / 1
                actual_ratio = max_uy[i] / max_uy[0]
                self.assertAlmostEqual(actual_ratio, expected_ratio, places=1,
                                       msg=f"Step {i}: ratio {actual_ratio:.2f} "
                                           f"vs expected {expected_ratio:.1f}")


if __name__ == "__main__":
    unittest.main()
