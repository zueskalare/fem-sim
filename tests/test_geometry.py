"""Tests for geometry generators."""

from __future__ import annotations

import unittest

import numpy as np

from fem_sim.geometry import (
    N_GEO_CHANNELS,
    CH_SOLID,
    CH_MATID,
    CH_E,
    CH_NU,
    make_rectangle,
    make_plate_with_hole,
    make_lshape,
    make_porous,
    make_bimat_rectangle,
    make_grf_bimat,
)


class TestGeometryGenerators(unittest.TestCase):
    """Test that geometry generators produce correct shapes and masks."""

    def test_rectangle_shape_and_full_solid(self):
        geo = make_rectangle(16, 32)
        self.assertEqual(geo.shape, (N_GEO_CHANNELS, 16, 32))
        np.testing.assert_array_equal(geo[CH_SOLID], 1.0)
        self.assertTrue(np.all(geo[CH_E] > 0))

    def test_plate_with_hole_has_void(self):
        geo = make_plate_with_hole(32, 32, cx=0.5, cy=0.5, r=0.15)
        self.assertEqual(geo.shape, (N_GEO_CHANNELS, 32, 32))
        n_void = np.sum(geo[CH_SOLID] < 0.5)
        self.assertGreater(n_void, 0, "Hole should create void pixels")
        void_mask = geo[CH_SOLID] < 0.5
        np.testing.assert_array_equal(geo[CH_E][void_mask], 0.0)

    def test_lshape_has_cutout(self):
        geo = make_lshape(32, 32, cut_frac=0.5)
        n_void = np.sum(geo[CH_SOLID] < 0.5)
        expected_void = 32 * 32 * 0.25
        self.assertAlmostEqual(n_void, expected_void, delta=2)

    def test_porous_has_holes(self):
        geo = make_porous(64, 64, n_pores=5, seed=42)
        n_void = np.sum(geo[CH_SOLID] < 0.5)
        self.assertGreater(n_void, 0, "Pores should create void pixels")
        solid_frac = np.sum(geo[CH_SOLID] > 0.5) / (64 * 64)
        self.assertGreater(solid_frac, 0.5)

    def test_bimat_two_materials(self):
        geo = make_bimat_rectangle(16, 32, split_frac=0.5, E1=210000, E2=70000)
        np.testing.assert_array_equal(geo[CH_SOLID], 1.0)
        self.assertAlmostEqual(geo[CH_E, 0, 0], 210000.0)
        self.assertAlmostEqual(geo[CH_E, 0, -1], 70000.0)

    def test_grf_bimat_two_phases(self):
        geo = make_grf_bimat(32, 64, correlation_length=4.0, volume_fraction=0.5, seed=0)
        self.assertEqual(geo.shape, (N_GEO_CHANNELS, 32, 64))
        # Fully solid — no voids, both phases are material.
        np.testing.assert_array_equal(geo[CH_SOLID], 1.0)
        # Exactly two material IDs: 1 (TPU) and 2 (PLA).
        np.testing.assert_array_equal(
            np.sort(np.unique(geo[CH_MATID])), np.array([1.0, 2.0])
        )
        # E field has exactly the two requested moduli.
        np.testing.assert_array_equal(
            np.sort(np.unique(geo[CH_E])), np.array([30.0, 3500.0])
        )

    def test_grf_bimat_is_deterministic(self):
        g1 = make_grf_bimat(32, 32, seed=7)
        g2 = make_grf_bimat(32, 32, seed=7)
        np.testing.assert_array_equal(g1, g2)
        # Different seed -> different mask.
        g3 = make_grf_bimat(32, 32, seed=8)
        self.assertFalse(np.array_equal(g1[CH_MATID], g3[CH_MATID]))

    def test_grf_bimat_volume_fraction(self):
        for target in (0.3, 0.5, 0.7):
            geo = make_grf_bimat(48, 96, volume_fraction=target, seed=1)
            frac_A = float(np.mean(geo[CH_MATID] == 1))
            self.assertAlmostEqual(frac_A, target, delta=0.02,
                                   msg=f"frac_A={frac_A:.3f} vs target={target}")


if __name__ == "__main__":
    unittest.main()
