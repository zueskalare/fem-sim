from __future__ import annotations

from pathlib import Path
import subprocess

from fem_sim.backends import register
from fem_sim.config import SimulationConfig, Scalar
from fem_sim.freefem_binary import find_freefem_binary
from fem_sim.result import RunResult

_BACKEND_NAME = "freefem"


@register(_BACKEND_NAME)
class FreeFemBackend:
    """Runs simulations via the FreeFEM+ external binary.

    Config mapping
    --------------
    outputs.*          ->  -key value           (direct, e.g. -run_dir, -field_file)
    mesh.*             ->  -mesh_key value
    solver.*           ->  -solver_key value
    parameters.*       ->  -param_key value
    backend_options.binary   ->  executable name (default "FreeFem++")
    backend_options.options  ->  list[str] inserted before the script
    backend_options.extra_args -> dict[str, str] appended as -key value after the script
    """

    def validate(self, config: SimulationConfig) -> list[str]:
        problems: list[str] = []
        try:
            find_freefem_binary(_binary_name(config))
        except FileNotFoundError:
            problems.append(f"FreeFEM+ binary not found (tried {_binary_name(config)!r})")
        script = _resolve_script(config)
        if script is None or not script.exists():
            problems.append(f"case_script not found: {config.case_script!r}")
        return problems

    def run(self, config: SimulationConfig) -> RunResult:
        binary = find_freefem_binary(_binary_name(config))
        script = _resolve_script(config)
        if script is None or not script.exists():
            raise FileNotFoundError(f"case_script not found: {config.case_script!r}")

        working_dir = _resolve_working_dir(config)
        _ensure_output_dirs(config)

        # -nw disables the GUI window; skip it when plot mode is requested
        plot_mode = bool(config.backend_options.get("plot", False))
        user_options: list[str] = list(config.backend_options.get("options", []))  # type: ignore[arg-type]
        options = list(user_options)
        if not plot_mode and "-nw" not in options:
            options.insert(0, "-nw")
        script_args = _build_script_args(config)
        extra = _flatten_extra_args(config.backend_options.get("extra_args", {}))  # type: ignore[arg-type]

        command = [binary, *options, str(script), *script_args, *extra]
        completed = subprocess.run(command, cwd=working_dir, check=False)

        return RunResult(
            command=command,
            working_directory=str(working_dir),
            return_code=completed.returncode,
            succeeded=completed.returncode == 0,
            backend=_BACKEND_NAME,
            outputs=_collect_outputs(config),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binary_name(config: SimulationConfig) -> str:
    return str(config.backend_options.get("binary", "FreeFem++"))


def _resolve_script(config: SimulationConfig) -> Path | None:
    path = Path(config.case_script)
    if path.is_absolute():
        return path
    # Try relative to cwd, then relative to project root env var (if set)
    if path.exists():
        return path.resolve()
    return None


def _resolve_working_dir(config: SimulationConfig) -> Path:
    if config.case_root:
        p = Path(config.case_root)
        return p.resolve() if not p.is_absolute() else p
    return Path.cwd()


def _ensure_output_dirs(config: SimulationConfig) -> None:
    """Create parent directories for all output paths before running FreeFEM.

    FreeFEM cannot create directories itself; missing parents cause silent
    write failures.  Checks both structured outputs and flat params (for
    run_dir / *_file keys that may come from the CLI or batch file).
    """
    candidates: list[str] = list(str(v) for v in config.outputs.values())
    for key, value in config.params.items():
        v = str(value)
        # Only treat values that look like paths (contain / or \, or end in known ext)
        if "/" in v or "\\" in v or Path(v).suffix:
            candidates.append(v)
        # run_dir specifically — always treat as a directory to create
        if key == "run_dir":
            Path(v).mkdir(parents=True, exist_ok=True)
            continue
    for v in candidates:
        Path(v).parent.mkdir(parents=True, exist_ok=True)


def _build_script_args(config: SimulationConfig) -> list[str]:
    """Translate config into FreeFEM CLI args.

    Priority (last wins for duplicate keys):
      structured sections (mesh/solver/parameters/outputs) → flat params
    """
    args: list[str] = []

    # outputs passed directly: -run_dir, -field_file, etc.
    for key, value in config.outputs.items():
        args.extend([f"-{key}", str(value)])

    # mesh / solver / parameters with section prefix
    for prefix, section in (
        ("mesh", config.mesh),
        ("solver", config.solver),
        ("param", config.parameters),
    ):
        args.extend(_flatten_section(prefix, section))

    # flat params — passed as -KEY VALUE with no prefix transformation
    for key, value in config.params.items():
        rendered = "1" if value is True else "0" if value is False else str(value)
        args.extend([f"-{key}", rendered])

    return args


def _flatten_section(prefix: str, values: dict[str, Scalar], stem: str = "") -> list[str]:
    args: list[str] = []
    for key, value in values.items():
        full_key = f"{stem}_{key}" if stem else key
        if isinstance(value, dict):
            args.extend(_flatten_section(prefix, value, full_key))  # type: ignore[arg-type]
            continue
        rendered = "1" if value is True else "0" if value is False else str(value)
        args.extend([f"-{prefix}_{full_key}", rendered])
    return args


def _flatten_extra_args(extra: dict[str, str]) -> list[str]:
    args: list[str] = []
    for key, value in extra.items():
        args.extend([f"-{key}", str(value)])
    return args


def _collect_outputs(config: SimulationConfig) -> dict[str, str]:
    """Return only the output paths that actually exist after the run."""
    result: dict[str, str] = {}
    for key, value in config.outputs.items():
        path = Path(str(value))
        if path.exists():
            result[key] = str(path.resolve())
    return result
