"""Tests for fem_sim.runner.run_fem (notebook-friendly orchestration wrapper)."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from fem_sim import backends as backends_mod
from fem_sim.config import SimulationConfig
from fem_sim.result import RunResult
from fem_sim.runner import run_fem


@dataclass
class _RecordingBackend:
    """Backend stub that captures every config it sees."""
    seen: list[SimulationConfig] = field(default_factory=list)
    return_code: int = 0
    problems: list[str] = field(default_factory=list)

    def validate(self, config: SimulationConfig) -> list[str]:
        return list(self.problems)

    def run(self, config: SimulationConfig) -> RunResult:
        self.seen.append(config)
        return RunResult(
            command=["stub", config.case_script],
            working_directory=".",
            return_code=self.return_code,
            succeeded=self.return_code == 0,
            backend="stub",
            outputs={},
        )


class _BackendPatch:
    """Context manager that inserts a stub backend under a name and removes it."""
    def __init__(self, name: str, backend: _RecordingBackend) -> None:
        self.name = name
        self.backend = backend
        self._saved: object = None

    def __enter__(self) -> _RecordingBackend:
        self._saved = backends_mod._REGISTRY.get(self.name)
        backends_mod._REGISTRY[self.name] = lambda: self.backend  # type: ignore[assignment]
        return self.backend

    def __exit__(self, *exc: object) -> None:
        if self._saved is None:
            backends_mod._REGISTRY.pop(self.name, None)
        else:
            backends_mod._REGISTRY[self.name] = self._saved  # type: ignore[assignment]


class TestRunFem(unittest.TestCase):
    def test_single_script_with_kwarg_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "fake.edp"
            script.write_text("// dummy\n")
            stub = _RecordingBackend()
            with _BackendPatch("stub", stub):
                results = run_fem(script, backend="stub", mesh_nx=32, dt=0.01)
        self.assertEqual(len(results), 1)
        self.assertEqual(stub.seen[0].params, {"mesh_nx": 32, "dt": 0.01})
        self.assertEqual(stub.seen[0].backend, "stub")

    def test_batch_from_json_with_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            script = tmp_p / "fake.edp"
            script.write_text("// dummy\n")
            batch = tmp_p / "batch.json"
            batch.write_text(json.dumps({
                "script": str(script),
                "runs": [{"run_id": "a", "x": 1}, {"run_id": "b", "x": 2}],
            }))
            stub = _RecordingBackend()
            with _BackendPatch("stub", stub):
                results = run_fem(params=batch, backend="stub", binary="/usr/bin/foo")
        self.assertEqual(len(results), 2)
        self.assertEqual([c.run_id for c in stub.seen], ["a", "b"])
        # binary override applied to every config
        self.assertTrue(all(c.backend_options.get("binary") == "/usr/bin/foo" for c in stub.seen))
        # batch entries become flat params
        self.assertEqual(stub.seen[0].params, {"x": 1})
        self.assertEqual(stub.seen[1].params, {"x": 2})

    def test_kwargs_merge_into_extra_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "fake.edp"
            script.write_text("// dummy\n")
            stub = _RecordingBackend()
            with _BackendPatch("stub", stub):
                run_fem(
                    script,
                    backend="stub",
                    extra_params={"mesh_nx": 16},
                    dt=0.05,  # kwarg
                )
        self.assertEqual(stub.seen[0].params, {"mesh_nx": 16, "dt": 0.05})

    def test_validation_failure_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "fake.edp"
            script.write_text("// dummy\n")
            stub = _RecordingBackend(problems=["mesh missing"])
            with _BackendPatch("stub", stub):
                with self.assertRaises(ValueError) as ctx:
                    run_fem(script, backend="stub")
        self.assertIn("mesh missing", str(ctx.exception))

    def test_either_script_or_params_required(self) -> None:
        with self.assertRaises(ValueError):
            run_fem()

    def test_script_and_params_are_mutually_exclusive(self) -> None:
        with self.assertRaises(ValueError):
            run_fem(script="x.edp", params="y.json")


if __name__ == "__main__":
    unittest.main()
