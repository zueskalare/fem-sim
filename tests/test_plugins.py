"""Tests for the geometry plugin system."""

from __future__ import annotations

import unittest

import numpy as np

from fem_sim.geometry import N_GEO_CHANNELS, CH_SOLID, CH_E, CH_NU, CH_RHO
from fem_sim.plugins import _REGISTRY, get_plugin, list_plugins, register_plugin
from fem_sim.plugins.base import GeoPlugin


class TestPluginRegistry(unittest.TestCase):
    """Registry mechanics: register, get, list, unknown."""

    def test_image2d_is_registered(self):
        names = list_plugins()
        self.assertIn("image2d", names)

    def test_get_plugin_returns_instance(self):
        plugin = get_plugin("image2d")
        self.assertIsInstance(plugin, GeoPlugin)

    def test_unknown_plugin_raises(self):
        with self.assertRaises(ValueError):
            get_plugin("nonexistent_plugin_xyz")

    def test_register_custom_plugin(self):
        @register_plugin("_test_dummy")
        class _Dummy:
            name = "dummy"
            def build(self, **kw):
                return np.zeros((5, 4, 4))
            def validate(self, **kw):
                return []

        self.assertIn("_test_dummy", list_plugins())
        p = get_plugin("_test_dummy")
        self.assertEqual(p.name, "dummy")
        # Clean up.
        del _REGISTRY["_test_dummy"]


class TestImage2DPlugin(unittest.TestCase):
    """Image2DPlugin with numpy array input (no Pillow needed)."""

    def setUp(self):
        self.plugin = get_plugin("image2d")
        # 16x32 solid block with a 4x8 void hole in the centre.
        self.mask = np.ones((16, 32), dtype=np.float64)
        self.mask[6:10, 12:20] = 0.0

    def test_build_shape(self):
        geo = self.plugin.build(image_array=self.mask)
        self.assertEqual(geo.shape, (N_GEO_CHANNELS, 16, 32))

    def test_solid_mask_matches_input(self):
        geo = self.plugin.build(image_array=self.mask)
        expected_solid = self.mask > 0.5
        np.testing.assert_array_equal(geo[CH_SOLID].astype(bool), expected_solid)

    def test_material_properties_on_solid(self):
        E, nu, rho = 100_000.0, 0.25, 5_000.0
        geo = self.plugin.build(image_array=self.mask, E=E, nu=nu, rho=rho)
        solid = geo[CH_SOLID].astype(bool)
        np.testing.assert_allclose(geo[CH_E][solid], E)
        np.testing.assert_allclose(geo[CH_NU][solid], nu)
        np.testing.assert_allclose(geo[CH_RHO][solid], rho)

    def test_void_pixels_are_zero(self):
        geo = self.plugin.build(image_array=self.mask)
        void = ~geo[CH_SOLID].astype(bool)
        self.assertTrue(np.all(geo[:, void] == 0.0))

    def test_default_material_properties(self):
        geo = self.plugin.build(image_array=np.ones((8, 8)))
        solid = geo[CH_SOLID].astype(bool)
        np.testing.assert_allclose(geo[CH_E][solid], 210_000.0)
        np.testing.assert_allclose(geo[CH_NU][solid], 0.3)
        np.testing.assert_allclose(geo[CH_RHO][solid], 7_800.0)


class TestImage2DThresholdInvert(unittest.TestCase):
    """Threshold and invert logic."""

    def setUp(self):
        self.plugin = get_plugin("image2d")

    def test_threshold(self):
        arr = np.array([[0.0, 0.4, 0.6, 1.0]])
        geo = self.plugin.build(image_array=arr, threshold=0.5)
        # After normalisation: [0, 0.4, 0.6, 1.0]; > 0.5 → [F, F, T, T]
        expected = [0.0, 0.0, 1.0, 1.0]
        np.testing.assert_array_equal(geo[CH_SOLID, 0, :], expected)

    def test_invert(self):
        arr = np.array([[0.0, 0.4, 0.6, 1.0]])
        geo = self.plugin.build(image_array=arr, threshold=0.5, invert=True)
        # Inverted: [T, T, F, F]
        expected = [1.0, 1.0, 0.0, 0.0]
        np.testing.assert_array_equal(geo[CH_SOLID, 0, :], expected)

    def test_uniform_image_all_solid(self):
        """A uniform image (vmax == vmin) should produce all-solid."""
        arr = np.full((4, 4), 128.0)
        geo = self.plugin.build(image_array=arr)
        np.testing.assert_array_equal(geo[CH_SOLID], 1.0)


class TestImage2DResize(unittest.TestCase):
    """Resize via target_h / target_w."""

    def setUp(self):
        self.plugin = get_plugin("image2d")

    def test_resize_both(self):
        arr = np.ones((100, 200))
        geo = self.plugin.build(image_array=arr, target_h=16, target_w=32)
        self.assertEqual(geo.shape, (N_GEO_CHANNELS, 16, 32))

    def test_resize_height_only(self):
        arr = np.ones((100, 50))
        geo = self.plugin.build(image_array=arr, target_h=20)
        self.assertEqual(geo.shape, (N_GEO_CHANNELS, 20, 50))

    def test_resize_width_only(self):
        arr = np.ones((30, 100))
        geo = self.plugin.build(image_array=arr, target_w=25)
        self.assertEqual(geo.shape, (N_GEO_CHANNELS, 30, 25))


class TestImage2DValidation(unittest.TestCase):
    """Validate method catches bad inputs."""

    def setUp(self):
        self.plugin = get_plugin("image2d")

    def test_no_source(self):
        problems = self.plugin.validate()
        self.assertTrue(any("image_path" in p or "image_array" in p for p in problems))

    def test_missing_file(self):
        problems = self.plugin.validate(image_path="/nonexistent/image.png")
        self.assertTrue(any("not found" in p for p in problems))

    def test_array_wrong_dims(self):
        problems = self.plugin.validate(image_array=np.ones((3, 4, 5)))
        self.assertTrue(any("2-D" in p for p in problems))

    def test_valid_array(self):
        problems = self.plugin.validate(image_array=np.ones((10, 10)))
        self.assertEqual(problems, [])


class TestCampaignIntegration(unittest.TestCase):
    """Plugin is reachable through campaign._build_geometry()."""

    def test_build_geometry_with_plugin(self):
        from fem_sim.campaign import _build_geometry
        mask = np.ones((12, 24))
        spec = {"type": "image2d", "image_array": mask, "E": 100_000.0}
        geo = _build_geometry(spec)
        self.assertEqual(geo.shape, (N_GEO_CHANNELS, 12, 24))

    def test_unknown_type_raises(self):
        from fem_sim.campaign import _build_geometry
        with self.assertRaises(ValueError):
            _build_geometry({"type": "totally_unknown_type_xyz"})


class TestDownstreamCompat(unittest.TestCase):
    """Plugin output works with the downstream pipeline."""

    def test_load_case_accepts_plugin_geometry(self):
        from fem_sim.load_case import make_cantilever_point_load
        plugin = get_plugin("image2d")
        geo = plugin.build(image_array=np.ones((16, 32)))
        bc = make_cantilever_point_load(geo)
        self.assertEqual(bc.shape[1:], (16, 32))


if __name__ == "__main__":
    unittest.main()
