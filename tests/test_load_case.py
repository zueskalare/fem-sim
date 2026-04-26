"""Tests for boundary condition / load case generators."""

from __future__ import annotations

import unittest

import numpy as np

from fem_sim.geometry import CH_SOLID, make_rectangle, make_plate_with_hole
from fem_sim.load_case import (
    N_BC_CHANNELS,
    CH_DISP_MASK,
    CH_FORCE_MASK,
    CH_DX,
    CH_DY,
    CH_FY,
    make_cantilever_point_load,
    make_cantilever_distributed,
    make_simply_supported_distributed,
    make_displacement_bc,
    make_top_load_fixed_bottom,
    make_uniaxial,
    make_shear,
    make_biaxial,
)


class TestLoadCaseGenerators(unittest.TestCase):
    """Test that BC generators produce valid tensors."""

    def setUp(self):
        self.geo = make_rectangle(16, 32)

    def test_cantilever_point_load_shape(self):
        bc = make_cantilever_point_load(self.geo, load_mag=-1000)
        self.assertEqual(bc.shape, (N_BC_CHANNELS, 16, 32))
        self.assertTrue(np.all(bc[CH_DISP_MASK, :, 0] == 1.0))
        n_force = np.sum(bc[CH_FORCE_MASK] > 0.5)
        self.assertEqual(n_force, 1)

    def test_cantilever_distributed_load(self):
        bc = make_cantilever_distributed(self.geo, load_mag=-500)
        right_force = bc[CH_FORCE_MASK, :, -1]
        self.assertTrue(np.all(right_force == 1.0))
        total_fy = np.sum(bc[CH_FY])
        self.assertAlmostEqual(total_fy, -500.0, places=3)

    def test_simply_supported(self):
        bc = make_simply_supported_distributed(self.geo, load_mag=-800)
        self.assertEqual(bc[CH_DISP_MASK, 0, 0], 1.0)
        self.assertEqual(bc[CH_DISP_MASK, 0, -1], 1.0)

    def test_displacement_bc(self):
        bc = make_displacement_bc(self.geo, disp_mag=0.01)
        self.assertTrue(np.all(bc[CH_DISP_MASK, :, 0] == 1.0))
        self.assertTrue(np.all(bc[CH_DISP_MASK, :, -1] == 1.0))

    def test_load_case_on_plate_with_hole(self):
        geo = make_plate_with_hole(32, 32, cx=0.5, cy=0.5, r=0.15)
        bc = make_cantilever_distributed(geo, load_mag=-1000)
        void = geo[CH_SOLID] < 0.5
        self.assertTrue(np.all(bc[CH_DISP_MASK][void] == 0.0))


class TestCharacterizationLoadCases(unittest.TestCase):
    """Uniaxial / shear / biaxial — square sample BC layouts."""

    def setUp(self):
        self.geo = make_rectangle(8, 8)  # square

    # -------- uniaxial --------

    def test_uniaxial_x_clamps_left_and_loads_right(self):
        bc = make_uniaxial(self.geo, disp_mag=0.05, direction="x")
        self.assertTrue(np.all(bc[CH_DISP_MASK, :, 0] == 1.0))    # left clamped
        self.assertTrue(np.all(bc[CH_DISP_MASK, :, -1] == 1.0))   # right loaded
        np.testing.assert_allclose(bc[CH_DX, :, 0], 0.0)          # left dx = 0
        np.testing.assert_allclose(bc[CH_DX, :, -1], 0.05)        # right dx = +0.05
        np.testing.assert_allclose(bc[CH_DY, :, -1], 0.0)         # right dy = 0
        # Top/bottom interior pixels untouched (only edge corners shared with left/right)
        self.assertTrue(np.all(bc[CH_DISP_MASK, 0, 1:-1] == 0.0))
        self.assertTrue(np.all(bc[CH_DISP_MASK, -1, 1:-1] == 0.0))

    def test_uniaxial_y_clamps_bottom_and_loads_top(self):
        bc = make_uniaxial(self.geo, disp_mag=-0.03, direction="y")  # compression
        self.assertTrue(np.all(bc[CH_DISP_MASK, 0, :] == 1.0))
        self.assertTrue(np.all(bc[CH_DISP_MASK, -1, :] == 1.0))
        np.testing.assert_allclose(bc[CH_DY, -1, :], -0.03)       # tension < 0
        np.testing.assert_allclose(bc[CH_DX, -1, :], 0.0)

    def test_uniaxial_invalid_direction_raises(self):
        with self.assertRaises(ValueError):
            make_uniaxial(self.geo, direction="z")

    # -------- shear --------

    def test_shear_x_top_slides_in_x(self):
        bc = make_shear(self.geo, disp_mag=0.02, direction="x")
        self.assertTrue(np.all(bc[CH_DISP_MASK, 0, :] == 1.0))    # bottom clamped
        self.assertTrue(np.all(bc[CH_DISP_MASK, -1, :] == 1.0))   # top sheared
        np.testing.assert_allclose(bc[CH_DX, -1, :], 0.02)
        np.testing.assert_allclose(bc[CH_DY, -1, :], 0.0)         # no thickness change

    def test_shear_y_right_slides_in_y(self):
        bc = make_shear(self.geo, disp_mag=0.02, direction="y")
        self.assertTrue(np.all(bc[CH_DISP_MASK, :, 0] == 1.0))
        self.assertTrue(np.all(bc[CH_DISP_MASK, :, -1] == 1.0))
        np.testing.assert_allclose(bc[CH_DY, :, -1], 0.02)
        np.testing.assert_allclose(bc[CH_DX, :, -1], 0.0)

    # -------- biaxial --------

    def test_biaxial_pins_bottom_left_and_loads_top_right(self):
        bc = make_biaxial(self.geo, disp_x=0.03, disp_y=0.05)
        # All four edges have disp_mask = 1
        for sl in [(slice(None), 0), (slice(None), -1), (0, slice(None)), (-1, slice(None))]:
            self.assertTrue(np.all(bc[CH_DISP_MASK][sl] == 1.0))

        # Bottom edge interior dx,dy = 0; right corner overwrites dx → disp_x.
        np.testing.assert_allclose(bc[CH_DX, 0, :-1], 0.0)
        np.testing.assert_allclose(bc[CH_DY, 0, :], 0.0)
        # Left edge interior dx,dy = 0; top corner overwrites dy → disp_y.
        np.testing.assert_allclose(bc[CH_DX, :, 0], 0.0)
        np.testing.assert_allclose(bc[CH_DY, :-1, 0], 0.0)
        # Top edge: dy = disp_y everywhere; dx = 0 except the right-corner overwrite.
        np.testing.assert_allclose(bc[CH_DY, -1, :], 0.05)
        np.testing.assert_allclose(bc[CH_DX, -1, :-1], 0.0)
        # Right edge: dx = disp_x everywhere; dy = 0 except the top-corner overwrite.
        np.testing.assert_allclose(bc[CH_DX, :, -1], 0.03)
        np.testing.assert_allclose(bc[CH_DY, :-1, -1], 0.0)
        # Four corners — natural biaxial layout.
        self.assertAlmostEqual(bc[CH_DX, 0, 0], 0.0)         # bottom-left pinned
        self.assertAlmostEqual(bc[CH_DY, 0, 0], 0.0)
        self.assertAlmostEqual(bc[CH_DX, 0, -1], 0.03)       # bottom-right pulled in x
        self.assertAlmostEqual(bc[CH_DY, 0, -1], 0.0)
        self.assertAlmostEqual(bc[CH_DX, -1, 0], 0.0)        # top-left pushed in y
        self.assertAlmostEqual(bc[CH_DY, -1, 0], 0.05)
        self.assertAlmostEqual(bc[CH_DX, -1, -1], 0.03)      # top-right both
        self.assertAlmostEqual(bc[CH_DY, -1, -1], 0.05)

    def test_biaxial_compression_uses_negative_signs(self):
        bc = make_biaxial(self.geo, disp_x=-0.01, disp_y=-0.02)
        self.assertAlmostEqual(bc[CH_DX, -1, -1], -0.01)
        self.assertAlmostEqual(bc[CH_DY, -1, -1], -0.02)
        # Mixed tension-compression also works
        bc = make_biaxial(self.geo, disp_x=0.01, disp_y=-0.02)
        self.assertAlmostEqual(bc[CH_DX, -1, -1], 0.01)
        self.assertAlmostEqual(bc[CH_DY, -1, -1], -0.02)


if __name__ == "__main__":
    unittest.main()
