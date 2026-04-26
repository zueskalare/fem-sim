from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

Scalar = str | int | float | bool


@dataclass
class SimulationConfig:
    """Describes a single simulation run.

    Fields
    ------
    case_script     Path to the solver script (.edp for FreeFEM, .py for FEniCSx, etc.)
    backend         Backend name: "freefem" | "fenicsx" | "abaqus" | "ansys"
    case_root       Working directory for the solver process (defaults to cwd)
    params          Flat key→value pairs passed directly as -KEY VALUE to the solver.
                    Use this for CLI runs and batch files.  No prefix is applied.
    mesh            Structured mesh params — forwarded as -mesh_KEY VALUE (FreeFEM legacy).
    solver          Structured solver params — forwarded as -solver_KEY VALUE.
    parameters      Structured physics params — forwarded as -param_KEY VALUE.
    outputs         Expected output paths forwarded as -KEY VALUE.
    backend_options Backend-specific options (binary path, PETSc options, etc.)
    run_id          Optional identifier used for labelling outputs / dataset index.
    case_name       Human-readable case name.
    """
    case_script: str
    backend: str = "freefem"
    case_root: str | None = None
    # Flat params — simplest way to pass args from CLI or batch file
    params: dict[str, Scalar] = field(default_factory=dict)
    # Structured sections — kept for backward compat with existing JSON configs
    mesh: dict[str, Scalar] = field(default_factory=dict)
    solver: dict[str, Scalar] = field(default_factory=dict)
    parameters: dict[str, Scalar] = field(default_factory=dict)
    outputs: dict[str, Scalar] = field(default_factory=dict)
    # Backend-specific options (binary, petsc_options, etc.)
    backend_options: dict[str, object] = field(default_factory=dict)
    run_id: str | None = None
    case_name: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "case_script": self.case_script,
            "backend": self.backend,
            "case_root": self.case_root,
            "params": self.params,
            "mesh": self.mesh,
            "solver": self.solver,
            "parameters": self.parameters,
            "outputs": self.outputs,
            "backend_options": self.backend_options,
            "run_id": self.run_id,
            "case_name": self.case_name,
        }


def read_dict_file(path: Path | str) -> dict:
    """Read a JSON or YAML file (auto-detected by extension) into a dict.

    Used by both single-config and campaign loaders so any code path
    accepts the same file formats.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load .yaml/.yml config files: "
                "install it with  uv add pyyaml"
            ) from exc
        return yaml.safe_load(text)
    return json.loads(text)


def load_config(path: Path | str) -> SimulationConfig:
    """Load a single SimulationConfig from a JSON or YAML file."""
    payload = read_dict_file(path)
    return _config_from_dict(payload)


def load_batch(path: Path | str) -> list[SimulationConfig]:
    """Load a batch of SimulationConfigs from a JSON (or YAML) file.

    Batch file format
    -----------------
    {
      "script":   "path/to/script.edp",   # required
      "backend":  "freefem",               # optional, default "freefem"
      "case_root": "...",                  # optional
      "backend_options": {...},            # optional, shared across all runs
      "runs": [
        {"run_id": "case_a", "mesh_nx": 16, "dt": 0.01},
        {"run_id": "case_b", "mesh_nx": 32, "dt": 0.005}
      ]
    }

    Each entry in "runs" is merged into a flat `params` dict.
    "run_id" is reserved and populates SimulationConfig.run_id instead.
    Any key in the top-level dict can be overridden per-run.
    """
    payload = read_dict_file(path)
    runs = payload.get("runs")
    if runs is None:
        # Single-config file — treat as load_config
        return [_config_from_dict(payload)]

    configs: list[SimulationConfig] = []
    for run in runs:
        # Merge top-level defaults with per-run overrides
        merged: dict[str, object] = {k: v for k, v in payload.items() if k != "runs"}
        merged.update(run)
        configs.append(_config_from_dict(merged))
    return configs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _config_from_dict(d: dict) -> SimulationConfig:
    """Build a SimulationConfig from a flat or structured dict.

    Keys that are not reserved config fields (case_script, backend, etc.)
    are collected into `params` automatically, so a batch run entry like
    {"mesh_nx": 16, "dt": 0.01} just works.
    """
    _RESERVED = {
        "case_script", "backend", "case_root", "params",
        "mesh", "solver", "parameters", "outputs",
        "backend_options", "run_id", "case_name",
        # legacy alias
        "script",
    }
    # Accept "script" as an alias for "case_script" (batch files)
    case_script = d.get("case_script") or d.get("script")
    if not case_script:
        raise ValueError("config must specify 'case_script' (or 'script' in batch files)")

    # Collect unknown keys as flat params
    extra_params: dict[str, Scalar] = {}
    for k, v in d.items():
        if k not in _RESERVED:
            extra_params[k] = v  # type: ignore[assignment]

    explicit_params: dict[str, Scalar] = dict(d.get("params", {}))  # type: ignore[arg-type]
    merged_params = {**extra_params, **explicit_params}  # explicit wins

    return SimulationConfig(
        case_script=str(case_script),
        backend=str(d.get("backend", "freefem")),
        case_root=d.get("case_root"),  # type: ignore[arg-type]
        params=merged_params,
        mesh=dict(d.get("mesh", {})),  # type: ignore[arg-type]
        solver=dict(d.get("solver", {})),  # type: ignore[arg-type]
        parameters=dict(d.get("parameters", {})),  # type: ignore[arg-type]
        outputs=dict(d.get("outputs", {})),  # type: ignore[arg-type]
        backend_options=dict(d.get("backend_options", {})),  # type: ignore[arg-type]
        run_id=d.get("run_id"),  # type: ignore[arg-type]
        case_name=d.get("case_name"),  # type: ignore[arg-type]
    )
