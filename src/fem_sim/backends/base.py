from __future__ import annotations

from typing import Protocol

from fem_sim.config import SimulationConfig
from fem_sim.result import RunResult


class BackendRunner(Protocol):
    """Protocol every FEM backend must satisfy."""

    def run(self, config: SimulationConfig) -> RunResult:
        """Execute the simulation and return the result."""
        ...

    def validate(self, config: SimulationConfig) -> list[str]:
        """Return a list of config problems.  Empty list means valid."""
        ...
