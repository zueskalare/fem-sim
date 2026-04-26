"""Tests for the jaxfem orchestration backend.

These tests don't require jax-fem to be installed — they use a fake
case_script whose solve() writes a marker file.  validate() is exercised
separately to confirm it reports jax-fem's absence when missing.
"""

from __future__ import annotations

import importlib.util
import tempfile
import textwrap
import unittest
from pathlib import Path

from fem_sim import get_backend, list_backends
from fem_sim.config import SimulationConfig


_FAKE_SOLVE_SCRIPT = textwrap.dedent(
    """
    from pathlib import Path

    def solve(config):
        out_dir = Path(config.outputs["run_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        marker = out_dir / "solved.txt"
        marker.write_text("ok")
        return {"marker": str(marker)}
    """
).strip()


_RAISING_SCRIPT = textwrap.dedent(
    """
    def solve(config):
        raise RuntimeError("boom")
    """
).strip()


_BAD_SCRIPT = textwrap.dedent(
    """
    # no solve function
    x = 1
    """
).strip()


_HAS_JAXFEM = importlib.util.find_spec("jax_fem") is not None


class TestJaxFemBackendRegistration(unittest.TestCase):
    def test_backend_registered(self) -> None:
        self.assertIn("jaxfem", list_backends())

    def test_get_backend(self) -> None:
        backend = get_backend("jaxfem")
        self.assertEqual(backend.__class__.__name__, "JaxFemBackend")


class TestJaxFemValidate(unittest.TestCase):
    def test_missing_script(self) -> None:
        cfg = SimulationConfig(case_script="/does/not/exist.py", backend="jaxfem")
        problems = get_backend("jaxfem").validate(cfg)
        self.assertTrue(any("case_script not found" in p for p in problems))

    def test_non_py_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "case.edp"
            script.write_text("// not python")
            cfg = SimulationConfig(case_script=str(script), backend="jaxfem")
            problems = get_backend("jaxfem").validate(cfg)
            self.assertTrue(any(".py file" in p for p in problems))

    @unittest.skipIf(_HAS_JAXFEM, "jax-fem is installed; skipping missing-dep check")
    def test_reports_missing_jax_fem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "case.py"
            script.write_text(_FAKE_SOLVE_SCRIPT)
            cfg = SimulationConfig(case_script=str(script), backend="jaxfem")
            problems = get_backend("jaxfem").validate(cfg)
            self.assertTrue(any("jax-fem" in p for p in problems))


class TestJaxFemRun(unittest.TestCase):
    def test_runs_solve_and_collects_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "case.py"
            script.write_text(_FAKE_SOLVE_SCRIPT)
            run_dir = Path(tmp) / "out"
            cfg = SimulationConfig(
                case_script=str(script),
                backend="jaxfem",
                outputs={"run_dir": str(run_dir)},
            )
            result = get_backend("jaxfem").run(cfg)

            self.assertTrue(result.succeeded)
            self.assertEqual(result.backend, "jaxfem")
            self.assertIn("marker", result.outputs)
            self.assertTrue(Path(result.outputs["marker"]).exists())

    def test_captures_solve_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "boom.py"
            script.write_text(_RAISING_SCRIPT)
            cfg = SimulationConfig(case_script=str(script), backend="jaxfem")
            result = get_backend("jaxfem").run(cfg)

        self.assertFalse(result.succeeded)
        self.assertNotEqual(result.return_code, 0)

    def test_missing_solve_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "bad.py"
            script.write_text(_BAD_SCRIPT)
            cfg = SimulationConfig(case_script=str(script), backend="jaxfem")
            with self.assertRaises(AttributeError):
                get_backend("jaxfem").run(cfg)


if __name__ == "__main__":
    unittest.main()
