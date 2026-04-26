"""Unified CLI for fem-sim: FEM simulation orchestration and dataset generation.

Subcommands:
    fem-sim run             Run one or more FEM simulations
    fem-sim index           Build a dataset index from completed runs
    fem-sim video-manifest  Build a video manifest for one run
    fem-sim build-dataset   Build a dataset campaign from config
    fem-sim inspect         Inspect a .npz dataset sample
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from fem_sim.backends import list_backends
from fem_sim.config import Scalar
from fem_sim.dataset import build_dataset_index
from fem_sim.runner import run_fem
from fem_sim.video import build_video_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FEM simulation orchestration and dataset generation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------ run
    run_p = subparsers.add_parser(
        "run",
        help="Run one or more FEM simulations.",
        description=(
            "Single run:  fem-sim run script.edp [KEY=VALUE ...]\n"
            "Batch run:   fem-sim run --params batch.json [KEY=VALUE ...]\n\n"
            "KEY=VALUE pairs are passed as -KEY VALUE to the solver script.\n"
            "Supported backends: " + ", ".join(list_backends())
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_p.add_argument(
        "script", type=Path, nargs="?",
        help="Solver script (.edp for FreeFEM, .py for FEniCSx, etc.)",
    )
    run_p.add_argument(
        "extra", nargs=argparse.REMAINDER,
        help="KEY=VALUE pairs forwarded to the solver as -KEY VALUE.",
    )
    run_p.add_argument(
        "--params", type=Path, default=None, metavar="FILE",
        help="JSON or YAML batch file.  Runs all parameter sets in 'runs' list.",
    )
    run_p.add_argument(
        "--backend", default=None,
        help="Override backend (freefem | fenicsx | abaqus | ansys).",
    )
    run_p.add_argument(
        "--binary", default=None,
        help="Override the solver binary name or path.",
    )
    run_p.add_argument(
        "--plot", action="store_true", default=False,
        help="Enable the solver GUI window for interactive plotting.",
    )

    # --------------------------------------------------------------- index
    idx_p = subparsers.add_parser("index", help="Build a dataset index from completed runs.")
    idx_p.add_argument("runs_root", type=Path)
    idx_p.add_argument("--output", type=Path, default=Path("outputs/datasets/index.json"))

    # -------------------------------------------------------- video-manifest
    vid_p = subparsers.add_parser("video-manifest", help="Build a video manifest for one run.")
    vid_p.add_argument("run_dir", type=Path)
    vid_p.add_argument("--output", type=Path, default=None)

    # -------------------------------------------------------- build-dataset
    build_p = subparsers.add_parser(
        "build-dataset",
        help="Build a dataset campaign from config.",
        description=(
            "Run a (geometry × load_case) sweep.  Use --limit to start with "
            "a small subset before committing to the full grid."
        ),
    )
    build_p.add_argument("--config", required=True, help="Path to campaign JSON or YAML config")
    build_p.add_argument(
        "--limit", type=int, default=None,
        help="Run only the first N pairs (after shuffling, if enabled).",
    )
    build_p.add_argument(
        "--shuffle", action="store_true",
        help="Randomize pair order before applying --limit.",
    )
    build_p.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for --shuffle (default: fresh randomness).",
    )
    build_p.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned sample IDs and exit without running anything.",
    )
    build_p.add_argument(
        "--export-vtk", action="store_true",
        help=(
            "After saving each sample, also export a time-series VTK collection "
            "(.vti per step + .pvd) under <output_dir>/vtk/<sample_id>/ "
            "for ParaView animation."
        ),
    )

    # -------------------------------------------------------- inspect
    insp_p = subparsers.add_parser("inspect", help="Inspect a .npz dataset sample.")
    insp_p.add_argument("npz_path", help="Path to .npz sample file")
    insp_p.add_argument("--step", type=int, default=None,
                        help="Load step to visualize (default: last)")

    # -------------------------------------------------------- export-vtk
    vtk_p = subparsers.add_parser(
        "export-vtk",
        help="Export a .npz sample as a time-series VTK collection (.vti per step + .pvd).",
    )
    vtk_p.add_argument("npz_path", type=Path, help="Path to .npz sample file")
    vtk_p.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (default: <sample_dir>/<stem>_vtk/).",
    )

    args = parser.parse_args()

    if args.command == "run":
        sys.exit(_cmd_run(args))
    elif args.command == "index":
        path = build_dataset_index(args.runs_root.resolve(), args.output.resolve())
        print(path)
    elif args.command == "video-manifest":
        output = args.output or (args.run_dir / "video_manifest.json")
        path = build_video_manifest(args.run_dir.resolve(), output.resolve())
        print(path)
    elif args.command == "build-dataset":
        _cmd_build_dataset(args)
    elif args.command == "inspect":
        _cmd_inspect(args)
    elif args.command == "export-vtk":
        _cmd_export_vtk(args)


def _cmd_run(args: argparse.Namespace) -> int:
    if args.params and args.script:
        print("error: use either a positional SCRIPT or --params FILE, not both.",
              file=sys.stderr)
        return 2
    if not args.params and not args.script:
        print("error: provide a solver SCRIPT or a --params FILE.", file=sys.stderr)
        return 2

    cli_params = _parse_kvpairs([a for a in (args.extra or []) if a != "--"])

    try:
        results = run_fem(
            script=args.script,
            params=args.params,
            backend=args.backend,
            binary=args.binary,
            plot=args.plot,
            extra_params=cli_params,
        )
    except ValueError as exc:
        # validate() failures bubble out as ValueError — print and exit 2.
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    worst_code = 0
    for result in results:
        label = Path(result.command[-1] if result.command else "?").stem
        print(f"ran [{label}] via {result.backend}: exit {result.return_code}")
        for key, path in result.outputs.items():
            print(f"  {key}: {path}")
        if not result.succeeded and result.return_code != 0:
            worst_code = result.return_code
    return worst_code


def _cmd_build_dataset(args: argparse.Namespace) -> None:
    import logging
    from fem_sim.campaign import build_dataset

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_dataset(
        args.config,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
        dry_run=args.dry_run,
        export_vtk=args.export_vtk,
    )


def _cmd_inspect(args: argparse.Namespace) -> None:
    from fem_sim.inspect import inspect
    inspect(args.npz_path, args.step)


def _cmd_export_vtk(args: argparse.Namespace) -> None:
    from fem_sim.vtk_export import export_sample_vtk
    pvd = export_sample_vtk(args.npz_path, output_dir=args.output)
    print(pvd)


def _parse_kvpairs(items: list[str]) -> dict[str, Scalar]:
    """Parse KEY=VALUE strings from the command line."""
    parsed: dict[str, Scalar] = {}
    for item in items:
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        key = key.lstrip("-").strip()
        parsed[key] = _cast(raw)
    return parsed


def _cast(value: str) -> Scalar:
    """Try to cast a string to int or float; fall back to str."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    return value
