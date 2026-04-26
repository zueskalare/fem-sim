"""Notebook-friendly wrapper for the ``fem-sim run`` orchestration subcommand.

Mirrors the CLI: take a script path or a batch params file, dispatch to the
chosen backend, return the list of ``RunResult`` objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from fem_sim.backends import get_backend
from fem_sim.config import Scalar, SimulationConfig, load_batch
from fem_sim.result import RunResult


def run_fem(
    script: str | Path | None = None,
    *,
    params: str | Path | None = None,
    backend: str | None = None,
    binary: str | None = None,
    plot: bool = False,
    extra_params: dict[str, Scalar] | None = None,
    backend_options: dict[str, object] | None = None,
    **kv: Scalar,
) -> list[RunResult]:
    """Run one or more FEM simulations through a backend.

    Mirrors ``fem-sim run [SCRIPT] [--params FILE] [KEY=VALUE...]``.

    Parameters
    ----------
    script : str or Path, optional
        Path to the solver script (``.edp`` for FreeFEM, ``.py`` for jaxfem).
        Mutually exclusive with ``params``.
    params : str or Path, optional
        Batch JSON / YAML file (with a ``runs:`` list).  Mutually exclusive
        with ``script``.
    backend : str, optional
        Override the backend on every loaded config (``"freefem"``,
        ``"jaxfem"``, etc.).
    binary : str, optional
        Override ``backend_options["binary"]`` on every config.
    plot : bool, default False
        Set ``backend_options["plot"]=True`` so FreeFEM keeps its GUI window.
    extra_params : dict, optional
        Flat ``-KEY VALUE`` overrides to merge into every config's ``params``.
    backend_options : dict, optional
        Extra ``backend_options`` to merge (alongside ``binary`` / ``plot``).
    **kv
        Convenience: extra ``-KEY VALUE`` pairs as Python kwargs.  Merged
        into ``extra_params`` (kwargs take precedence on key conflicts).

    Returns
    -------
    list[RunResult]
        One result per executed config (single-element when ``script`` is used).

    Examples
    --------
    >>> # Single FreeFEM run
    >>> results = run_fem("script.edp", mesh_nx=32, dt=0.01)
    >>> # Batch sweep from a JSON file, force jaxfem backend
    >>> results = run_fem(params="batch.json", backend="jaxfem")
    """
    if script is None and params is None:
        raise ValueError("provide either a script path or a params batch file")
    if script is not None and params is not None:
        raise ValueError("script and params are mutually exclusive")

    if params is not None:
        configs = load_batch(Path(params).resolve())
    else:
        configs = [SimulationConfig(case_script=str(Path(script).resolve()))]

    merged_extras: dict[str, Scalar] = {}
    if extra_params:
        merged_extras.update(extra_params)
    merged_extras.update(kv)

    for cfg in configs:
        if backend:
            cfg.backend = backend
        if binary:
            cfg.backend_options = {**cfg.backend_options, "binary": binary}
        if plot:
            cfg.backend_options = {**cfg.backend_options, "plot": True}
        if backend_options:
            cfg.backend_options = {**cfg.backend_options, **backend_options}
        if merged_extras:
            cfg.params = {**cfg.params, **merged_extras}

    return [_run_one(cfg) for cfg in configs]


def _run_one(config: SimulationConfig) -> RunResult:
    backend = get_backend(config.backend)
    problems = backend.validate(config)
    if problems:
        raise ValueError(
            f"config validation failed for {config.case_script!r}: {problems}"
        )
    return backend.run(config)
