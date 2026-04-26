"""JAX-FEM backend.

The case_script must be a Python file exposing a top-level callable

    solve(config: SimulationConfig) -> dict[str, str] | None

where the returned dict maps output keys to filesystem paths.  Keys from
the returned dict are merged on top of config.outputs when building the
RunResult, so the solver can report paths it chose at runtime.

jax-fem is optional; install with ``uv sync --extra jaxfem``.
"""

from __future__ import annotations

import importlib.util as _ilu
import sys
import traceback
from pathlib import Path

from fem_sim.backends import register
from fem_sim.config import SimulationConfig
from fem_sim.result import RunResult

_BACKEND_NAME = "jaxfem"
_SOLVE_ATTR = "solve"


@register(_BACKEND_NAME)
class JaxFemBackend:
    """Runs simulations via the JAX-FEM Python library.

    Config mapping
    --------------
    case_script        Python file with a top-level ``solve(config)`` function.
    case_root          Working directory for the solver (defaults to cwd).
    outputs            Seed map of output paths; solver may override via its
                       return value.
    backend_options    Passed through unchanged — solver decides what to use.
    """

    def validate(self, config: SimulationConfig) -> list[str]:
        problems: list[str] = []
        if _ilu.find_spec("jax_fem") is None:
            problems.append(
                "jax-fem is not installed; run `uv sync --extra jaxfem`"
            )
        script = _resolve_script(config)
        if script is None or not script.exists():
            problems.append(f"case_script not found: {config.case_script!r}")
            return problems
        if script.suffix != ".py":
            problems.append(
                f"jaxfem case_script must be a .py file (got {script.suffix!r})"
            )
        return problems

    def run(self, config: SimulationConfig) -> RunResult:
        script = _resolve_script(config)
        if script is None or not script.exists():
            raise FileNotFoundError(f"case_script not found: {config.case_script!r}")

        working_dir = _resolve_working_dir(config)
        _ensure_output_dirs(config)

        module = _load_module(script)
        solve = getattr(module, _SOLVE_ATTR, None)
        if not callable(solve):
            raise AttributeError(
                f"{script} must define a top-level '{_SOLVE_ATTR}(config)' function"
            )

        command = ["python", str(script)]
        return_code = 0
        outputs: dict[str, str] = {str(k): str(v) for k, v in config.outputs.items()}

        import os
        cwd = Path.cwd()
        os.chdir(working_dir)
        try:
            result = solve(config)
        except Exception:
            traceback.print_exc()
            return_code = 1
            result = None
        finally:
            os.chdir(cwd)

        if isinstance(result, dict):
            outputs.update({str(k): str(v) for k, v in result.items()})

        outputs = {k: str(Path(v).resolve()) for k, v in outputs.items() if Path(v).exists()}

        return RunResult(
            command=command,
            working_directory=str(working_dir),
            return_code=return_code,
            succeeded=return_code == 0,
            backend=_BACKEND_NAME,
            outputs=outputs,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_script(config: SimulationConfig) -> Path | None:
    path = Path(config.case_script)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return None


def _resolve_working_dir(config: SimulationConfig) -> Path:
    if config.case_root:
        p = Path(config.case_root)
        return p.resolve() if not p.is_absolute() else p
    return Path.cwd()


def _ensure_output_dirs(config: SimulationConfig) -> None:
    for value in config.outputs.values():
        v = str(value)
        if "/" in v or "\\" in v or Path(v).suffix:
            Path(v).parent.mkdir(parents=True, exist_ok=True)


def _load_module(script: Path):
    mod_name = f"_fem_sim_jaxfem_case_{script.stem}_{abs(hash(str(script)))}"
    spec = _ilu.spec_from_file_location(mod_name, script)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {script}")
    module = _ilu.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module
