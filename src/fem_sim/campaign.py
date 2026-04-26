"""Dataset campaign builder: orchestrates geometry x load_case -> FEM -> .npz.

Can be driven programmatically or from a JSON / YAML config file.

Usage (CLI):
    fem-sim build-dataset --config configs/grf_bimat_campaign.yaml

Usage (Python):
    from fem_sim.campaign import build_campaign, CampaignConfig
    config = CampaignConfig.from_file("configs/grf_bimat_campaign.yaml")
    build_campaign(config)

Config schema (YAML shown; same keys for JSON):

    output_dir: outputs/datasets/campaign_01
    steps: 10
    backend: jaxfem            # or "freefem"

    materials:                 # optional named library
      TPU:   {E: 30.0,    nu: 0.48, rho: 1.2e-9}
      PLA:   {E: 3500.0,  nu: 0.36, rho: 1.24e-9}
      steel: {E: 210000,  nu: 0.3,  rho: 7800}

    geometries:
      - type: rectangle
        h: 32
        w: 64
        materials: steel        # scalar shorthand for single-slot generators
      - type: grf_bimat
        h: 48
        w: 96
        correlation_length: 4.0
        volume_fraction: 0.5
        materials: [TPU, PLA]   # list fills slots (E_A,nu_A,rho_A) (E_B,nu_B,rho_B)

    load_cases:
      - {type: cantilever_point_load, load_mag: -1000}
      - {type: cantilever_distributed, load_mag: -500}

The ``materials`` field on a geometry spec is optional — literal ``E``,
``nu``, ``rho`` (or ``E_A``, ``nu_A``, …) keys still work.  When both are
present, the literal value wins.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from fem_sim import geometry as geo_mod
from fem_sim import load_case as lc_mod
from fem_sim.config import read_dict_file
from fem_sim.pixel_to_fem import FEMSample, run_simulation, save_sample

logger = logging.getLogger(__name__)

_MATERIAL_PROPS = ("E", "nu", "rho")
# Recognises material slot suffixes like "", "_A", "_B", "1", "2", "_metal".
_E_PARAM_RE = re.compile(r"^E(_[A-Za-z]+|\d+)?$")

# Registry of geometry generators keyed by type name.
_GEO_GENERATORS = {
    "rectangle": geo_mod.make_rectangle,
    "plate_with_hole": geo_mod.make_plate_with_hole,
    "lshape": geo_mod.make_lshape,
    "porous": geo_mod.make_porous,
    "bimat_rectangle": geo_mod.make_bimat_rectangle,
    "grf_bimat": geo_mod.make_grf_bimat,
}

# Registry of load case generators keyed by type name.
_LC_GENERATORS = {
    "cantilever_point_load": lc_mod.make_cantilever_point_load,
    "cantilever_distributed": lc_mod.make_cantilever_distributed,
    "simply_supported_distributed": lc_mod.make_simply_supported_distributed,
    "displacement_bc": lc_mod.make_displacement_bc,
    "top_load_fixed_bottom": lc_mod.make_top_load_fixed_bottom,
    "uniaxial": lc_mod.make_uniaxial,
    "shear": lc_mod.make_shear,
    "biaxial": lc_mod.make_biaxial,
}


@dataclass
class CampaignConfig:
    """Configuration for a dataset generation campaign.

    The ``materials`` library maps a name to ``{E, nu, rho}``.  Geometry
    specs reference materials by name via a ``materials:`` field — see
    module docstring for syntax.
    """

    output_dir: Path
    geometries: list[dict[str, Any]]
    load_cases: list[dict[str, Any]]
    steps: int = 10
    backend: str = "freefem"
    freefem_binary: str | None = None
    materials: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> CampaignConfig:
        """Load a campaign config from a JSON or YAML file."""
        data = read_dict_file(path)
        return cls(
            output_dir=Path(data["output_dir"]),
            geometries=data["geometries"],
            load_cases=data["load_cases"],
            steps=data.get("steps", 10),
            backend=data.get("backend", "freefem"),
            freefem_binary=data.get("freefem_binary"),
            materials=dict(data.get("materials", {})),
        )

    # JSON-only alias retained for callers that only handle .json paths.
    from_json = from_file


def _material_slots(fn: Callable[..., Any]) -> list[str]:
    """Return the ``E*``-style suffixes of a generator's material slots.

    For ``make_rectangle(..., E, nu, rho)`` returns ``[""]``;
    for ``make_grf_bimat(..., E_A, ..., E_B, ...)`` returns ``["_A", "_B"]``;
    for ``make_bimat_rectangle(..., E1, ..., E2, ...)`` returns ``["1", "2"]``.

    A slot is only included when matching ``nu{suffix}`` and ``rho{suffix}``
    parameters also exist.
    """
    params = inspect.signature(fn).parameters
    slots: list[str] = []
    for name in params:
        m = _E_PARAM_RE.match(name)
        if not m:
            continue
        suffix = m.group(1) or ""
        if f"nu{suffix}" in params and f"rho{suffix}" in params:
            slots.append(suffix)
    return slots


def _resolve_materials(
    spec: dict[str, Any],
    materials: dict[str, dict[str, float]],
    slots: list[str],
    geo_type: str,
) -> dict[str, Any]:
    """Expand ``materials:`` references in *spec* into literal E/nu/rho kwargs.

    The expanded dict is returned (input is not mutated).  Literal ``E``,
    ``nu``, ``rho`` keys already in the spec take precedence over material
    references — this is how a user can override one slot of a bi-material
    spec while still pulling the other from the library.
    """
    out = {k: v for k, v in spec.items() if k not in ("type", "materials")}
    refs = spec.get("materials")
    if refs is None:
        return out

    if isinstance(refs, str):
        refs = [refs]
    if not isinstance(refs, list):
        raise ValueError(
            f"geometry {geo_type!r}: 'materials' must be a name or a list of names, "
            f"got {type(refs).__name__}"
        )
    if len(refs) != len(slots):
        raise ValueError(
            f"geometry {geo_type!r} expects {len(slots)} material(s) for "
            f"slot suffix(es) {slots!r}, got {len(refs)}: {refs!r}"
        )

    for slot, ref in zip(slots, refs):
        if ref not in materials:
            raise ValueError(
                f"unknown material {ref!r} in geometry {geo_type!r}; "
                f"defined: {sorted(materials)}"
            )
        props = materials[ref]
        for prop in _MATERIAL_PROPS:
            key = f"{prop}{slot}"
            if key in out:
                continue  # literal in spec wins
            if prop not in props:
                raise ValueError(
                    f"material {ref!r} missing required property {prop!r}"
                )
            out[key] = props[prop]
    return out


def _build_geometry(
    spec: dict[str, Any],
    materials: dict[str, dict[str, float]] | None = None,
) -> np.ndarray:
    """Build a geometry array from a spec dict.

    Checks built-in generators first, then falls back to the plugin registry.
    """
    geo_type = spec["type"]
    materials = materials or {}

    if geo_type in _GEO_GENERATORS:
        fn = _GEO_GENERATORS[geo_type]
        kwargs = _resolve_materials(spec, materials, _material_slots(fn), geo_type)
        return fn(**kwargs)

    # Plugin path — plugins don't expose introspectable material slots, so
    # we don't auto-resolve.  Plugins can read 'materials' from kwargs themselves.
    from fem_sim.plugins import get_plugin, list_plugins
    try:
        plugin = get_plugin(geo_type)
    except ValueError:
        raise ValueError(
            f"Unknown geometry type: {geo_type!r}. "
            f"Built-in: {list(_GEO_GENERATORS)}. "
            f"Plugins: {list_plugins()}"
        )
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    problems = plugin.validate(**kwargs)
    if problems:
        raise ValueError(f"Plugin {geo_type!r} validation failed: {problems}")
    return plugin.build(**kwargs)


def _build_load_case(spec: dict[str, Any], geo: np.ndarray) -> np.ndarray:
    """Build a BC array from a spec dict and geometry."""
    lc_type = spec["type"]
    if lc_type not in _LC_GENERATORS:
        raise ValueError(f"Unknown load case type: {lc_type}. "
                         f"Available: {list(_LC_GENERATORS)}")
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    return _LC_GENERATORS[lc_type](geo, **kwargs)


def _select_pairs(
    n_geo: int,
    n_lc: int,
    limit: int | None,
    shuffle: bool,
    seed: int | None,
) -> list[tuple[int, int]]:
    """Return the (gi, li) pairs that should run, after shuffle + limit."""
    pairs = [(gi, li) for gi in range(n_geo) for li in range(n_lc)]
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(pairs)
    if limit is not None:
        pairs = pairs[:limit]
    return pairs


def build_campaign(
    config: CampaignConfig,
    limit: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    dry_run: bool = False,
    export_vtk: bool = False,
) -> list[Path]:
    """Run a (geometry × load_case) sweep through FEM.

    By default every combination runs.  Use the sampling parameters to
    pull a small subset first when you don't want to wait for the full
    dataset:

    Parameters
    ----------
    limit : int, optional
        Run only the first ``limit`` (gi, li) pairs (after shuffling, if
        enabled).  Useful for sanity-checking a config before committing
        to a long run.
    shuffle : bool, default False
        Shuffle pair order before applying ``limit``.  Pairs are selected
        from the full ``n_geometries × n_load_cases`` grid.
    seed : int, optional
        RNG seed for ``shuffle``.  Omit for fresh randomness; pass an
        integer for reproducible subsets.
    dry_run : bool, default False
        Log the planned sample IDs and return without running anything.
    export_vtk : bool, default False
        After saving each sample, also export a time-series VTK collection
        (one ``.vti`` per load step + ``.pvd``) under
        ``<output_dir>/vtk/<sample_id>/`` for ParaView animation.

    Returns
    -------
    list[Path]
        ``.npz`` paths for successfully completed samples (empty for dry
        runs and for runs where every pair failed).
    """
    pairs = _select_pairs(
        len(config.geometries), len(config.load_cases),
        limit=limit, shuffle=shuffle, seed=seed,
    )
    total_in_grid = len(config.geometries) * len(config.load_cases)

    if dry_run:
        logger.info(
            "Dry run: %d/%d pairs would build (limit=%s, shuffle=%s, seed=%s)",
            len(pairs), total_in_grid, limit, shuffle, seed,
        )
        for gi, li in pairs:
            geo_type = config.geometries[gi].get("type", "?")
            lc_type = config.load_cases[li].get("type", "?")
            logger.info("  g%03d × l%02d : %s + %s", gi, li, geo_type, lc_type)
        return []

    config.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[Path] = []

    for gi, li in pairs:
        geo_spec = config.geometries[gi]
        lc_spec = config.load_cases[li]

        geo = _build_geometry(geo_spec, config.materials)
        bc = _build_load_case(lc_spec, geo)

        geo_label = f"{geo_spec['type']}_{geo.shape[1]}x{geo.shape[2]}"
        lc_label = lc_spec["type"]
        # Sample ID is keyed on (gi, li) so a sampled run and a full run
        # produce identical IDs for any overlapping pair.
        sample_id = f"g{gi:03d}_l{li:02d}_{geo_label}_{lc_label}"
        run_dir = config.output_dir / "runs" / sample_id
        npz_path = config.output_dir / "samples" / f"{sample_id}.npz"

        logger.info("Sample %s: %s + %s", sample_id, geo_label, lc_label)

        try:
            sample = run_simulation(
                geo, bc,
                steps=config.steps,
                run_dir=run_dir,
                freefem_binary=config.freefem_binary,
                backend=config.backend,
            )
            sample.metadata["geometry_spec"] = geo_spec
            sample.metadata["load_case_spec"] = lc_spec
            sample.metadata["sample_id"] = sample_id
            sample.metadata["geometry_index"] = gi
            sample.metadata["load_case_index"] = li
            save_sample(sample, npz_path)
            results.append(npz_path)
            logger.info("  -> saved %s", npz_path)

            if export_vtk:
                # Imported lazily so users not exporting VTK don't pay the import cost.
                from fem_sim.vtk_export import export_sample_vtk
                vtk_dir = config.output_dir / "vtk" / sample_id
                pvd = export_sample_vtk(sample, output_dir=vtk_dir, name=sample_id)
                logger.info("  -> vtk %s", pvd)
        except Exception:
            logger.exception("  -> FAILED %s", sample_id)

    index = {
        "total_samples": len(results),
        "total_attempted": len(pairs),
        "total_in_grid": total_in_grid,
        "limit": limit,
        "shuffle": shuffle,
        "seed": seed,
        "export_vtk": export_vtk,
        "samples": [str(p) for p in results],
    }
    index_path = config.output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    logger.info(
        "Campaign complete: %d/%d samples (subset %d/%d). Index: %s",
        len(results), len(pairs), len(pairs), total_in_grid, index_path,
    )

    return results


def build_dataset(
    config: str | Path | CampaignConfig,
    *,
    output_dir: str | Path | None = None,
    steps: int | None = None,
    backend: str | None = None,
    materials: dict[str, dict[str, float]] | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    dry_run: bool = False,
    export_vtk: bool = False,
) -> list[Path]:
    """Run a dataset campaign — notebook-friendly wrapper.

    Equivalent to ``CampaignConfig.from_file(path) + build_campaign(...)`` but
    in one call.  Optional keyword args override the corresponding fields on
    the loaded config (handy for trying a smaller ``steps`` or a different
    ``output_dir`` without editing the YAML).

    Parameters
    ----------
    config : str, Path, or CampaignConfig
        Path to a JSON / YAML campaign config, or an already-loaded
        ``CampaignConfig`` instance.
    output_dir, steps, backend, materials : optional
        Override the loaded config's matching field.  ``None`` means "keep".
    limit, shuffle, seed, dry_run, export_vtk : optional
        Forwarded to :func:`build_campaign`; see its docstring for sampling
        semantics and VTK export behaviour.

    Returns
    -------
    list[Path]
        ``.npz`` paths for successfully completed samples (empty for dry runs).

    Examples
    --------
    >>> # Notebook one-liner — sample first 4 pairs from a YAML config
    >>> paths = build_dataset("configs/grf_bimat_campaign.yaml", limit=4)
    >>> # Override output_dir and steps without editing the file
    >>> paths = build_dataset(
    ...     "configs/grf_bimat_campaign.yaml",
    ...     output_dir="outputs/notebook_run",
    ...     steps=3,
    ...     limit=2,
    ... )
    """
    if isinstance(config, (str, Path)):
        config = CampaignConfig.from_file(config)

    overrides: dict[str, Any] = {}
    if output_dir is not None:
        overrides["output_dir"] = Path(output_dir)
    if steps is not None:
        overrides["steps"] = steps
    if backend is not None:
        overrides["backend"] = backend
    if materials is not None:
        overrides["materials"] = materials
    if overrides:
        config = replace(config, **overrides)

    return build_campaign(
        config,
        limit=limit, shuffle=shuffle, seed=seed,
        dry_run=dry_run, export_vtk=export_vtk,
    )


def main() -> None:
    """CLI entry point."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Build FEM dataset campaign")
    parser.add_argument("--config", required=True, help="Path to campaign JSON or YAML config")
    args = parser.parse_args()

    config = CampaignConfig.from_file(args.config)
    build_campaign(config)


if __name__ == "__main__":
    main()
