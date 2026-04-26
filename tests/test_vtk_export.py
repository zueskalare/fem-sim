"""Tests for fem_sim.vtk_export — VTI per step + PVD collection."""

from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from fem_sim import (
    export_sample_vtk,
    make_cantilever_distributed,
    make_rectangle,
)
from fem_sim.pixel_to_fem import FEMSample, save_sample


def _fake_sample(h: int = 8, w: int = 8, steps: int = 3) -> FEMSample:
    """Build a small in-memory sample with deterministic field values."""
    geo = make_rectangle(h, w)
    bc = make_cantilever_distributed(geo, load_mag=-1.0)
    # 5 field channels: ux, uy, sxx, syy, sxy.  Inject a recognisable pattern
    # per step so we can verify cell ordering on read-back.
    fields = np.zeros((steps, 5, h, w), dtype=np.float64)
    for t in range(steps):
        for c in range(5):
            fields[t, c] = (t + 1) * (c + 1) * 0.001
    return FEMSample(geometry=geo, boundary=bc, fields=fields, metadata={})


def _read_data_array(piece: ET.Element, name: str) -> np.ndarray:
    arr = next(a for a in piece.find("CellData").findall("DataArray") if a.get("Name") == name)
    return np.fromstring(arr.text.strip(), sep=" ")


class TestExportSampleVtk(unittest.TestCase):
    def test_in_memory_sample_writes_pvd_plus_vti_per_step(self):
        sample = _fake_sample(h=4, w=8, steps=3)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            pvd = export_sample_vtk(sample, output_dir=tmp_p, name="my_sample")

            self.assertTrue(pvd.exists())
            self.assertEqual(pvd.name, "my_sample.pvd")
            self.assertEqual(sorted(p.name for p in tmp_p.glob("step_*.vti")),
                             ["step_0000.vti", "step_0001.vti", "step_0002.vti"])

    def test_pvd_collection_lists_every_step_with_load_fraction(self):
        sample = _fake_sample(steps=4)
        with tempfile.TemporaryDirectory() as tmp:
            pvd = export_sample_vtk(sample, output_dir=tmp, name="x")
            root = ET.parse(pvd).getroot()
            ds = root.findall(".//DataSet")
            self.assertEqual(len(ds), 4)
            timesteps = [float(d.get("timestep")) for d in ds]
            # Load fractions: 1/4, 2/4, 3/4, 4/4
            np.testing.assert_allclose(timesteps, [0.25, 0.5, 0.75, 1.0])
            # File references are relative paths so the bundle is portable
            for d in ds:
                self.assertFalse(Path(d.get("file")).is_absolute())

    def test_vti_extent_matches_pixel_grid(self):
        sample = _fake_sample(h=4, w=12, steps=2)
        with tempfile.TemporaryDirectory() as tmp:
            export_sample_vtk(sample, output_dir=tmp, name="x")
            root = ET.parse(Path(tmp) / "step_0000.vti").getroot()
            extent = root.find("ImageData").get("WholeExtent")
            self.assertEqual(extent, "0 12 0 4 0 0")  # "0 W 0 H 0 0"

    def test_vti_carries_all_expected_arrays_with_correct_arity(self):
        sample = _fake_sample(h=4, w=4, steps=1)
        with tempfile.TemporaryDirectory() as tmp:
            export_sample_vtk(sample, output_dir=tmp, name="x")
            root = ET.parse(Path(tmp) / "step_0000.vti").getroot()
            piece = root.find("ImageData").find("Piece")
            arrays = {a.get("Name"): a.get("NumberOfComponents") for a in piece.find("CellData").findall("DataArray")}
            # Static channels
            for name in ("solid_mask", "material_id", "E", "nu", "rho",
                         "disp_mask", "force_mask"):
                self.assertIn(name, arrays)
                self.assertIsNone(arrays[name])  # scalar → no NumberOfComponents
            # Vectors
            for name in ("prescribed_disp", "prescribed_force", "displacement"):
                self.assertEqual(arrays.get(name), "2")
            # Time-varying scalars
            for name in ("stress_xx", "stress_yy", "stress_xy"):
                self.assertIn(name, arrays)

    def test_displacement_values_match_source_at_each_step(self):
        sample = _fake_sample(h=4, w=4, steps=3)
        with tempfile.TemporaryDirectory() as tmp:
            export_sample_vtk(sample, output_dir=tmp, name="x")
            for t in range(3):
                root = ET.parse(Path(tmp) / f"step_{t:04d}.vti").getroot()
                piece = root.find("ImageData").find("Piece")
                disp = _read_data_array(piece, "displacement")
                # 4*4 cells, 2 components (ux, uy) interleaved
                self.assertEqual(disp.size, 4 * 4 * 2)
                disp_2d = disp.reshape(4, 4, 2)
                # ux comes from fields[t, 0] = (t+1) * 1 * 0.001
                expected_ux = (t + 1) * 1 * 0.001
                expected_uy = (t + 1) * 2 * 0.001
                np.testing.assert_allclose(disp_2d[..., 0], expected_ux)
                np.testing.assert_allclose(disp_2d[..., 1], expected_uy)

    def test_path_input_loads_npz_and_picks_default_output_dir(self):
        sample = _fake_sample(h=4, w=4, steps=2)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            npz = tmp_p / "my_sample.npz"
            save_sample(sample, npz)

            pvd = export_sample_vtk(npz)  # no output_dir
            # Default is <sample_dir>/<stem>_vtk/
            self.assertEqual(pvd.parent, tmp_p / "my_sample_vtk")
            self.assertEqual(pvd.name, "my_sample.pvd")
            self.assertEqual(len(list((tmp_p / "my_sample_vtk").glob("step_*.vti"))), 2)

    def test_in_memory_sample_requires_output_dir(self):
        sample = _fake_sample(steps=1)
        with self.assertRaises(ValueError):
            export_sample_vtk(sample)


if __name__ == "__main__":
    unittest.main()
