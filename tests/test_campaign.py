"""Tests for fem_sim.campaign — config loading + materials library."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fem_sim import geometry as geo_mod
from fem_sim.campaign import (
    CampaignConfig,
    _build_geometry,
    _material_slots,
    _partition_pairs,
    _resolve_materials,
    _select_pairs,
    build_campaign,
    build_dataset,
)


class TestMaterialSlots(unittest.TestCase):
    """Verify slot suffix introspection on the built-in generators."""

    def test_single_slot_generators(self) -> None:
        for fn in (
            geo_mod.make_rectangle,
            geo_mod.make_plate_with_hole,
            geo_mod.make_lshape,
            geo_mod.make_porous,
        ):
            self.assertEqual(_material_slots(fn), [""], f"{fn.__name__}")

    def test_bimat_rectangle_slots(self) -> None:
        self.assertEqual(_material_slots(geo_mod.make_bimat_rectangle), ["1", "2"])

    def test_grf_bimat_slots(self) -> None:
        self.assertEqual(_material_slots(geo_mod.make_grf_bimat), ["_A", "_B"])


class TestResolveMaterials(unittest.TestCase):
    MATS = {
        "steel": {"E": 210000.0, "nu": 0.30, "rho": 7.8e-9},
        "TPU":   {"E": 30.0,     "nu": 0.48, "rho": 1.2e-9},
        "PLA":   {"E": 3500.0,   "nu": 0.36, "rho": 1.24e-9},
    }

    def test_no_materials_field_passes_through(self) -> None:
        spec = {"type": "rectangle", "h": 8, "w": 16, "E": 1.0}
        out = _resolve_materials(spec, self.MATS, [""], "rectangle")
        self.assertEqual(out, {"h": 8, "w": 16, "E": 1.0})

    def test_scalar_shorthand_for_single_slot(self) -> None:
        spec = {"type": "rectangle", "h": 8, "w": 16, "materials": "steel"}
        out = _resolve_materials(spec, self.MATS, [""], "rectangle")
        self.assertAlmostEqual(out["E"], 210000.0)
        self.assertAlmostEqual(out["nu"], 0.30)
        self.assertAlmostEqual(out["rho"], 7.8e-9)
        # type and materials keys are stripped
        self.assertNotIn("type", out)
        self.assertNotIn("materials", out)

    def test_list_fills_two_slots_in_order(self) -> None:
        spec = {"type": "grf_bimat", "h": 8, "w": 8, "materials": ["TPU", "PLA"]}
        out = _resolve_materials(spec, self.MATS, ["_A", "_B"], "grf_bimat")
        self.assertAlmostEqual(out["E_A"], 30.0)
        self.assertAlmostEqual(out["E_B"], 3500.0)
        self.assertAlmostEqual(out["nu_A"], 0.48)
        self.assertAlmostEqual(out["nu_B"], 0.36)

    def test_literal_in_spec_overrides_material_ref(self) -> None:
        spec = {
            "type": "grf_bimat", "h": 8, "w": 8,
            "materials": ["TPU", "PLA"],
            "E_A": 99.0,  # explicit override
        }
        out = _resolve_materials(spec, self.MATS, ["_A", "_B"], "grf_bimat")
        self.assertAlmostEqual(out["E_A"], 99.0)
        self.assertAlmostEqual(out["E_B"], 3500.0)  # PLA still used for B

    def test_unknown_material_name_raises(self) -> None:
        spec = {"type": "rectangle", "h": 8, "w": 16, "materials": "missing"}
        with self.assertRaises(ValueError) as ctx:
            _resolve_materials(spec, self.MATS, [""], "rectangle")
        self.assertIn("missing", str(ctx.exception))

    def test_wrong_arity_raises(self) -> None:
        spec = {"type": "grf_bimat", "h": 8, "w": 8, "materials": ["TPU"]}
        with self.assertRaises(ValueError) as ctx:
            _resolve_materials(spec, self.MATS, ["_A", "_B"], "grf_bimat")
        self.assertIn("expects 2", str(ctx.exception))

    def test_material_missing_property_raises(self) -> None:
        bad = {"missing_rho": {"E": 1.0, "nu": 0.3}}
        spec = {"type": "rectangle", "h": 8, "w": 16, "materials": "missing_rho"}
        with self.assertRaises(ValueError) as ctx:
            _resolve_materials(spec, bad, [""], "rectangle")
        self.assertIn("rho", str(ctx.exception))


class TestBuildGeometryWithMaterials(unittest.TestCase):
    def test_rectangle_with_material_ref(self) -> None:
        spec = {"type": "rectangle", "h": 4, "w": 8, "materials": "steel"}
        materials = {"steel": {"E": 210000.0, "nu": 0.30, "rho": 7.8e-9}}
        geo = _build_geometry(spec, materials)
        self.assertEqual(geo.shape, (5, 4, 8))
        # All solid pixels should have steel's E.
        np.testing.assert_allclose(geo[geo_mod.CH_E], 210000.0)
        np.testing.assert_allclose(geo[geo_mod.CH_NU], 0.30)

    def test_grf_bimat_with_material_refs(self) -> None:
        spec = {
            "type": "grf_bimat",
            "h": 8, "w": 8, "seed": 0,
            "correlation_length": 2.0, "volume_fraction": 0.5,
            "materials": ["TPU", "PLA"],
        }
        materials = {
            "TPU": {"E": 30.0,   "nu": 0.48, "rho": 1.2e-9},
            "PLA": {"E": 3500.0, "nu": 0.36, "rho": 1.24e-9},
        }
        geo = _build_geometry(spec, materials)
        # Two distinct E values present, matching the two materials.
        e_unique = sorted(np.unique(geo[geo_mod.CH_E]))
        self.assertEqual(len(e_unique), 2)
        np.testing.assert_allclose(e_unique, [30.0, 3500.0])

    def test_inline_E_nu_rho_still_works(self) -> None:
        """Backwards-compat path: explicit literals, no materials library."""
        spec = {"type": "rectangle", "h": 4, "w": 8, "E": 1.0, "nu": 0.25, "rho": 1.0}
        geo = _build_geometry(spec)
        np.testing.assert_allclose(geo[geo_mod.CH_E], 1.0)


class TestCampaignConfigLoading(unittest.TestCase):
    def test_load_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "campaign.yaml"
            path.write_text(
                "output_dir: outputs/x\n"
                "steps: 3\n"
                "backend: jaxfem\n"
                "materials:\n"
                "  steel: {E: 210000.0, nu: 0.3, rho: 7.8e-9}\n"
                "geometries:\n"
                "  - {type: rectangle, h: 4, w: 8, materials: steel}\n"
                "load_cases:\n"
                "  - {type: cantilever_distributed, load_mag: -1.0}\n"
            )
            cfg = CampaignConfig.from_file(path)
        self.assertEqual(cfg.steps, 3)
        self.assertEqual(cfg.backend, "jaxfem")
        self.assertEqual(cfg.output_dir, Path("outputs/x"))
        self.assertEqual(list(cfg.materials), ["steel"])
        self.assertAlmostEqual(cfg.materials["steel"]["E"], 210000.0)

    def test_load_json_still_works(self) -> None:
        payload = {
            "output_dir": "outputs/x",
            "steps": 3,
            "geometries": [{"type": "rectangle", "h": 4, "w": 8, "E": 1.0}],
            "load_cases": [{"type": "cantilever_distributed", "load_mag": -1.0}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "campaign.json"
            path.write_text(json.dumps(payload))
            cfg = CampaignConfig.from_file(path)
        self.assertEqual(cfg.materials, {})  # absent → empty


class TestSelectPairs(unittest.TestCase):
    def test_full_grid_when_no_limit(self) -> None:
        pairs = _select_pairs(3, 2, limit=None, shuffle=False, seed=None)
        self.assertEqual(pairs, [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])

    def test_limit_takes_first_n_deterministically(self) -> None:
        pairs = _select_pairs(3, 3, limit=4, shuffle=False, seed=None)
        self.assertEqual(pairs, [(0, 0), (0, 1), (0, 2), (1, 0)])

    def test_limit_larger_than_grid_returns_full_grid(self) -> None:
        pairs = _select_pairs(2, 2, limit=99, shuffle=False, seed=None)
        self.assertEqual(len(pairs), 4)

    def test_shuffle_with_seed_is_reproducible(self) -> None:
        a = _select_pairs(4, 4, limit=5, shuffle=True, seed=42)
        b = _select_pairs(4, 4, limit=5, shuffle=True, seed=42)
        c = _select_pairs(4, 4, limit=5, shuffle=True, seed=43)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_shuffle_preserves_pair_set(self) -> None:
        full = set(_select_pairs(3, 3, limit=None, shuffle=False, seed=None))
        shuffled = set(_select_pairs(3, 3, limit=None, shuffle=True, seed=0))
        self.assertEqual(full, shuffled)


class TestPartitionPairs(unittest.TestCase):
    """Round-robin pair partition for MPI dispatch."""

    def test_single_rank_returns_all(self) -> None:
        pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.assertEqual(_partition_pairs(pairs, rank=0, size=1), pairs)

    def test_round_robin_split_two_ranks(self) -> None:
        pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.assertEqual(_partition_pairs(pairs, rank=0, size=2), [(0, 0), (1, 0)])
        self.assertEqual(_partition_pairs(pairs, rank=1, size=2), [(0, 1), (1, 1)])

    def test_partition_is_disjoint_and_complete(self) -> None:
        pairs = [(g, l) for g in range(5) for l in range(3)]  # 15 pairs
        for size in (1, 2, 3, 4, 7, 16):
            chunks = [_partition_pairs(pairs, r, size) for r in range(size)]
            flat = [p for c in chunks for p in c]
            self.assertEqual(sorted(flat), sorted(pairs), f"size={size}")
            # No overlap between any two ranks.
            for r in range(size):
                for s in range(r + 1, size):
                    self.assertEqual(
                        set(chunks[r]) & set(chunks[s]),
                        set(),
                        f"size={size}, ranks {r},{s}",
                    )

    def test_uneven_split_load_balanced(self) -> None:
        # 10 pairs across 4 ranks → counts (3, 3, 2, 2), not (3, 3, 3, 1).
        pairs = list(range(10))
        counts = [len(_partition_pairs(pairs, r, 4)) for r in range(4)]
        self.assertEqual(sorted(counts, reverse=True), [3, 3, 2, 2])


class TestBuildCampaignSampling(unittest.TestCase):
    """End-to-end sampling checks using a stub backend (no FEM solve)."""

    def _stub_config(self, tmp: Path) -> CampaignConfig:
        return CampaignConfig(
            output_dir=tmp,
            geometries=[
                {"type": "rectangle", "h": 4, "w": 8, "E": 1.0, "nu": 0.3, "rho": 1.0},
                {"type": "rectangle", "h": 4, "w": 8, "E": 2.0, "nu": 0.3, "rho": 1.0},
                {"type": "rectangle", "h": 4, "w": 8, "E": 3.0, "nu": 0.3, "rho": 1.0},
            ],
            load_cases=[
                {"type": "cantilever_distributed", "load_mag": -1.0},
                {"type": "cantilever_distributed", "load_mag": -2.0},
            ],
            steps=2,
        )

    def test_dry_run_runs_no_simulations_and_creates_no_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._stub_config(Path(tmp))
            results = build_campaign(cfg, limit=2, dry_run=True)
            self.assertEqual(results, [])
            # Dry run does not create the output dir or any files.
            self.assertFalse((Path(tmp) / "samples").exists())
            self.assertFalse((Path(tmp) / "runs").exists())
            self.assertFalse((Path(tmp) / "index.json").exists())

    def test_limit_produces_subset_of_samples(self) -> None:
        # Patch run_simulation so we don't need a FEM backend in this test.
        from fem_sim import campaign as campaign_mod
        from fem_sim.pixel_to_fem import FEMSample

        def fake_run(geo, bc, steps, run_dir, freefem_binary, backend):
            run_dir.mkdir(parents=True, exist_ok=True)
            return FEMSample(
                geometry=geo, boundary=bc,
                fields=np.zeros((steps, 5, geo.shape[1], geo.shape[2])),
                metadata={},
            )

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._stub_config(Path(tmp))
            real = campaign_mod.run_simulation
            campaign_mod.run_simulation = fake_run
            try:
                results = build_campaign(cfg, limit=2)
            finally:
                campaign_mod.run_simulation = real

            self.assertEqual(len(results), 2)
            # Sample IDs are stable on (gi, li): first two pairs are (0, 0) and (0, 1).
            ids = sorted(p.stem for p in results)
            self.assertTrue(ids[0].startswith("g000_l00_"))
            self.assertTrue(ids[1].startswith("g000_l01_"))

            index = json.loads((Path(tmp) / "index.json").read_text())
            self.assertEqual(index["total_samples"], 2)
            self.assertEqual(index["total_in_grid"], 6)
            self.assertEqual(index["limit"], 2)
            self.assertFalse(index["export_vtk"])  # off by default
            # mpi_size present even in single-process runs (size=1).
            self.assertEqual(index["mpi_size"], 1)

    def test_export_vtk_writes_per_sample_collections(self) -> None:
        # Reuse the same stub backend pattern.
        from fem_sim import campaign as campaign_mod
        from fem_sim.pixel_to_fem import FEMSample

        def fake_run(geo, bc, steps, run_dir, freefem_binary, backend):
            run_dir.mkdir(parents=True, exist_ok=True)
            return FEMSample(
                geometry=geo, boundary=bc,
                fields=np.zeros((steps, 5, geo.shape[1], geo.shape[2])),
                metadata={},
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            cfg = self._stub_config(tmp_p)
            real = campaign_mod.run_simulation
            campaign_mod.run_simulation = fake_run
            try:
                results = build_campaign(cfg, limit=2, export_vtk=True)
            finally:
                campaign_mod.run_simulation = real

            self.assertEqual(len(results), 2)
            # vtk/ dir parallels samples/ and runs/
            vtk_root = tmp_p / "vtk"
            self.assertTrue(vtk_root.is_dir())
            self.assertEqual(sorted(p.name for p in vtk_root.iterdir()),
                             [results[0].stem, results[1].stem])

            # Each per-sample dir has steps + a .pvd named after the sample
            for npz in results:
                sample_vtk = vtk_root / npz.stem
                self.assertEqual(len(list(sample_vtk.glob("step_*.vti"))), cfg.steps)
                self.assertTrue((sample_vtk / f"{npz.stem}.pvd").exists())

            index = json.loads((tmp_p / "index.json").read_text())
            self.assertTrue(index["export_vtk"])


class TestBuildDatasetWrapper(unittest.TestCase):
    """Notebook-friendly wrapper that loads config and forwards to build_campaign."""

    def _write_yaml(self, tmp: Path) -> Path:
        path = tmp / "campaign.yaml"
        path.write_text(
            "output_dir: outputs/from_yaml\n"
            "steps: 5\n"
            "backend: jaxfem\n"
            "geometries:\n"
            "  - {type: rectangle, h: 4, w: 8, E: 1.0, nu: 0.3, rho: 1.0}\n"
            "  - {type: rectangle, h: 4, w: 8, E: 2.0, nu: 0.3, rho: 1.0}\n"
            "load_cases:\n"
            "  - {type: cantilever_distributed, load_mag: -1.0}\n"
            "  - {type: cantilever_distributed, load_mag: -2.0}\n"
        )
        return path

    def test_accepts_path_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_yaml(Path(tmp))
            res = build_dataset(str(path), dry_run=True)
        self.assertEqual(res, [])

    def test_accepts_already_loaded_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_yaml(Path(tmp))
            cfg = CampaignConfig.from_file(path)
            res = build_dataset(cfg, dry_run=True, limit=2)
        self.assertEqual(res, [])

    def test_overrides_apply_without_mutating_loaded_config(self) -> None:
        # Verify that overrides don't affect the cached config object.
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_yaml(Path(tmp))
            cfg = CampaignConfig.from_file(path)
            original_steps = cfg.steps
            original_dir = cfg.output_dir

            build_dataset(cfg, output_dir=Path(tmp) / "alt", steps=99, dry_run=True)

            # Loaded config object is unchanged because build_dataset uses dataclasses.replace.
            self.assertEqual(cfg.steps, original_steps)
            self.assertEqual(cfg.output_dir, original_dir)

    def test_forwards_sampling_kwargs(self) -> None:
        # Patch run_simulation so the test doesn't need a real backend.
        from fem_sim import campaign as campaign_mod
        from fem_sim.pixel_to_fem import FEMSample

        def fake_run(geo, bc, steps, run_dir, freefem_binary, backend):
            run_dir.mkdir(parents=True, exist_ok=True)
            return FEMSample(
                geometry=geo, boundary=bc,
                fields=np.zeros((steps, 5, geo.shape[1], geo.shape[2])),
                metadata={},
            )

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_yaml(Path(tmp))
            real = campaign_mod.run_simulation
            campaign_mod.run_simulation = fake_run
            try:
                results = build_dataset(
                    path,
                    output_dir=Path(tmp) / "out",
                    steps=2,
                    limit=3,
                )
            finally:
                campaign_mod.run_simulation = real

            self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
