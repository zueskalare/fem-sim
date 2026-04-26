"""Integration tests that run FreeFEM EDP scripts through fem-sim.

Each test invokes ``fem-sim run`` on a real EDP file under
``sim/freefem/examples/`` and asserts that the solver exits successfully.
The tests are skipped automatically when FreeFem++ is not installed.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "sim" / "freefem" / "examples"

_HAS_FREEFEM = (
    shutil.which("FreeFem++") is not None
    or list(Path("/Applications").glob("FreeFem++.app/Contents/ff-*/bin/FreeFem++"))
)

_skip_reason = "FreeFem++ not installed"


def _run_edp(script: Path, extra_args: list[str] | None = None) -> subprocess.CompletedProcess[bytes]:
    """Run a single EDP script through fem-sim and return the result."""
    cmd = ["uv", "run", "fem-sim", "run", str(script)]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, timeout=120)


@unittest.skipUnless(_HAS_FREEFEM, _skip_reason)
class TestFreeFEMTutorial(unittest.TestCase):
    """Tests for scripts under sim/freefem/examples/tutorial/."""

    def test_laplace(self) -> None:
        result = _run_edp(EXAMPLES_DIR / "tutorial" / "Laplace.edp")
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_convect(self) -> None:
        result = _run_edp(EXAMPLES_DIR / "tutorial" / "convect.edp")
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_adapt(self) -> None:
        result = _run_edp(EXAMPLES_DIR / "tutorial" / "adapt.edp")
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_func(self) -> None:
        result = _run_edp(EXAMPLES_DIR / "tutorial" / "func.edp")
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_array(self) -> None:
        result = _run_edp(EXAMPLES_DIR / "tutorial" / "array.edp")
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_newton(self) -> None:
        result = _run_edp(EXAMPLES_DIR / "tutorial" / "Newton.edp")
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_ns_backward_step(self) -> None:
        result = _run_edp(EXAMPLES_DIR / "tutorial" / "NS-BackwardStep.edp")
        self.assertEqual(result.returncode, 0, result.stderr.decode())


@unittest.skipUnless(_HAS_FREEFEM, _skip_reason)
class TestFreeFEMMisc(unittest.TestCase):
    """Tests for scripts under sim/freefem/examples/misc/."""

    def test_demo(self) -> None:
        result = _run_edp(EXAMPLES_DIR / "misc" / "demo.edp")
        self.assertEqual(result.returncode, 0, result.stderr.decode())


@unittest.skipUnless(_HAS_FREEFEM, _skip_reason)
class TestFreeFEMFrameworkSmoke(unittest.TestCase):
    """Smoke test using the framework_smoke.edp with params."""

    def test_framework_smoke_with_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "smoke_run"
            result = _run_edp(
                EXAMPLES_DIR / "framework_smoke.edp",
                extra_args=[
                    f"run_dir={run_dir}",
                    f"series_file={run_dir}/series.tsv",
                    f"mesh_file={run_dir}/mesh.mesh",
                    f"field_file={run_dir}/trajectory.vtu",
                    f"summary_file={run_dir}/summary.tsv",
                    "mesh_nx=8",
                    "mesh_ny=8",
                    "solver_steps=4",
                    "solver_dt=0.1",
                    "param_decay=0.25",
                ],
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertTrue((run_dir / "series.tsv").exists(), "series.tsv not created")
            self.assertTrue((run_dir / "mesh.mesh").exists(), "mesh.mesh not created")
            self.assertTrue((run_dir / "summary.tsv").exists(), "summary.tsv not created")


if __name__ == "__main__":
    unittest.main()
