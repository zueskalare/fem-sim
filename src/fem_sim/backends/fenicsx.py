# FenicsxBackend — runs simulations via the FEniCSx Python API (dolfinx).
#
# Not yet implemented.  dolfinx must be installed separately (not in pyproject.toml).
# The registry in backends/__init__.py silently skips this backend if dolfinx is absent.
#
# When implemented:
#   - case_script points to a Python module exposing run(mesh, solver, parameters, outputs)
#   - mesh/solver/parameters dicts forwarded as kwargs
#   - outputs.run_dir used as XDMF/VTK output directory
#   - backend_options.petsc_options forwarded to PETSc solver
import importlib.util as _ilu

if _ilu.find_spec("dolfinx") is None:
    raise ImportError("dolfinx is not installed; fenicsx backend unavailable")

from fem_sim.backends import register  # noqa: E402
from fem_sim.config import SimulationConfig  # noqa: E402
from fem_sim.result import RunResult  # noqa: E402


@register("fenicsx")
class FenicsxBackend:
    def validate(self, config: SimulationConfig) -> list[str]:
        return ["fenicsx backend is not yet implemented"]

    def run(self, config: SimulationConfig) -> RunResult:
        raise NotImplementedError("fenicsx backend is not yet implemented")
