"""Tests for SimulationConfig loading and round-trip."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from fem_sim.config import load_config, load_batch


class TestSimulationConfig(unittest.TestCase):

    def test_load_config_reads_structured_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps({
                    "case_script": "sim/freefem/examples/framework_smoke.edp",
                    "mesh": {"nx": 8, "ny": 6},
                    "solver": {"steps": 4, "dt": 0.1},
                    "parameters": {"decay": 0.2},
                    "outputs": {"field_file": "outputs/run/trajectory.vtu"},
                }),
                encoding="utf-8",
            )
            config = load_config(config_path)
            self.assertEqual(config.mesh["nx"], 8)
            self.assertEqual(config.solver["dt"], 0.1)
            self.assertEqual(config.parameters["decay"], 0.2)
            self.assertEqual(config.outputs["field_file"], "outputs/run/trajectory.vtu")

    def test_case_root_round_trips_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps({
                    "case_script": "sim/freefem/examples/framework_smoke.edp",
                    "case_root": "sim/freefem/cases",
                }),
                encoding="utf-8",
            )
            config = load_config(config_path)
            self.assertEqual(config.case_root, "sim/freefem/cases")

    def test_load_batch_creates_multiple_configs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "batch.json"
            config_path.write_text(
                json.dumps({
                    "script": "solver.edp",
                    "runs": [
                        {"run_id": "a", "mesh_nx": 8},
                        {"run_id": "b", "mesh_nx": 16},
                    ],
                }),
                encoding="utf-8",
            )
            configs = load_batch(config_path)
            self.assertEqual(len(configs), 2)
            self.assertEqual(configs[0].run_id, "a")
            self.assertEqual(configs[0].params["mesh_nx"], 8)
            self.assertEqual(configs[1].run_id, "b")
            self.assertEqual(configs[1].params["mesh_nx"], 16)

    def test_to_dict_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps({
                    "case_script": "test.edp",
                    "backend": "freefem",
                    "params": {"nx": 10},
                }),
                encoding="utf-8",
            )
            config = load_config(config_path)
            d = config.to_dict()
            self.assertEqual(d["case_script"], "test.edp")
            self.assertEqual(d["params"]["nx"], 10)


if __name__ == "__main__":
    unittest.main()
