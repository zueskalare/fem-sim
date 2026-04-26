# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`fem-sim` is a standalone Python package that combines FEM simulation orchestration with pixel-geometry-driven dataset generation for ML training. It produces `.npz` dataset samples consumed by the companion `fem-world-model` package (at `../prj-8-video_model_FEM_pred/`).

## Current Status

**Last Updated**: 2026-04-25
**Stage**: Active Development
**Summary**: Multi-backend FEM (FreeFEM + JAX-FEM) with a GRF bi-material dataset pipeline. Campaign configs accept JSON or YAML, materials are pulled from a named library, sampling subsets supported via `--limit`/`--shuffle`. Every CLI subcommand has a matching Python wrapper (`build_dataset`, `run_fem`, `inspect`, `export_sample_vtk`, …) so notebooks share code with the CLI. Characterization load cases (uniaxial / shear / biaxial) added for square bimaterial samples. ParaView animation export via `.vti` per step + `.pvd` collection — both standalone (`fem-sim export-vtk`) and inline with dataset generation (`build-dataset --export-vtk`). 120 tests pass (1 auto-skipped when `jax-fem` is missing).

## Work Log

### 2026-04-25
- **VTK export inline with dataset generation.** Added `export_vtk` kwarg to `build_campaign` and `build_dataset`; `--export-vtk` CLI flag on `fem-sim build-dataset`. When set, each successful sample also writes a `.vti`-per-step + `.pvd` collection under `<output_dir>/vtk/<sample_id>/`, parallel to `samples/` and `runs/`. Choice is recorded in `index.json`. +1 test → **120 total**.
- **VTK time-series export.** New `src/fem_sim/vtk_export.py` with `export_sample_vtk(sample_or_path, output_dir=None, name=None) -> Path` that writes one `.vti` (VTK ImageData) per load step plus a `.pvd` collection that ParaView opens as an animation.  Pixel grid → ImageData (uniform spacing 1, no mesh/cell connectivity), CellData arrays for every channel; `displacement` and the prescribed BCs are written as 2-component vectors so ParaView can warp by displacement.  Hand-rolled XML — no `vtk`/`pyvista` dep.  CLI: `fem-sim export-vtk <sample.npz> [--output DIR]`; default output is `<sample_dir>/<stem>_vtk/`.  Top-level re-export.  7 tests in `tests/test_vtk_export.py`.
- **Characterization load cases.** Added `make_uniaxial(geo, disp_mag, direction)`, `make_shear(geo, disp_mag, direction)`, `make_biaxial(geo, disp_x, disp_y)` in `load_case.py`. Registered in `_LC_GENERATORS` as `uniaxial`/`shear`/`biaxial`; re-exported from package root. Same `direction='x'|'y'` pattern as existing generators. Engineering sign convention (positive disp → tension). New `configs/bimat_characterization_campaign.yaml`: 3 GRF microstructures × 8 modes (uniaxial x±, uniaxial y±, shear x, shear y, biaxial tension, biaxial compression) = 24 samples. `make_biaxial` corner handling: layered edge writes naturally produce the right four-corner displacements (bottom-left=(0,0), bottom-right=(disp_x,0), top-left=(0,disp_y), top-right=(disp_x,disp_y)) — no explicit corner code needed. 7 new tests in `test_load_case.py` → **112 total**.
- **Notebook-friendly CLI wrappers.** Added `fem_sim.build_dataset(config, *, output_dir, steps, backend, materials, limit, shuffle, seed, dry_run)` (in `campaign.py`) and `fem_sim.run_fem(script, *, params, backend, binary, plot, extra_params, **kv)` (new `runner.py`).  Each mirrors a CLI subcommand 1:1.  `_cmd_run` and `_cmd_build_dataset` in `cli.py` were refactored to delegate to these wrappers, so CLI and notebook share one code path.  Top-level re-exports added: `build_dataset`, `run_fem`, `inspect`.  Tests: `tests/test_runner.py` (6) + new wrapper tests in `test_campaign.py` (4) → **105 total**.
- **Code review pass.** Fixed `backends/freefem.py` `-nw`/plot ternary bug; rewrote `dataset.py` (was scanning for nonexistent `.vtu`/`.xdmf`) to scan `.npz` samples + sidecar JSON; rewrote `video.py` to find `fields_step_*.tsv` sorted by integer step. `inspect.py` now plots all 6 BC channels (was 5/6, missing `fy`).
- **Vectorized hot loops.** `pixel_to_fem.{write_geometry_dat,write_boundary_dat,load_field_tsv}` and `solvers/elasticity2d_jaxfem.py` (read/write helpers, `_sample_displacement`, `_sample_stress`) now use `np.savetxt` / `np.loadtxt` / fancy indexing instead of per-pixel Python loops.
- **Quieter physics knobs.** JAX-FEM solver clamps `nu` to `(0, 0.499]` with a logged warning instead of silently rewriting `nu=0.5 → 0.3`.
- **Refactored `load_case.py`.** Four `_*_edge_mask` helpers collapsed into `_solid_edge(geo, edge)`; five distributed-load generators share `_apply_distributed_load`.
- **YAML campaign configs.** `read_dict_file` (in `config.py`) auto-detects `.yaml`/`.yml`/`.json`. `pyyaml>=6.0` is now a core dep. `CampaignConfig.from_file` is the canonical loader (`from_json` retained as alias).
- **Materials library.** `CampaignConfig.materials: dict[str, {E, nu, rho}]`. Each geometry spec can now reference materials via a `materials:` field — scalar (`materials: steel`) for single-slot generators, list (`materials: [TPU, PLA]`) for multi-slot. Slot inference is signature-based: any `E*`/`nu*`/`rho*` parameter group (suffix `""`, `"_A"`, `"1"`, etc.) is auto-detected. Literal `E`/`nu`/`rho` keys still work and override the matching slot.
- **Sample subset support.** `build_campaign(config, limit, shuffle, seed, dry_run)` and CLI flags `--limit N`, `--shuffle`, `--seed S`, `--dry-run`. Sample IDs renamed `g{gi:03d}_l{li:02d}_<labels>` so they're stable across `--limit` and full runs (overlapping pairs share filenames).
- New config `configs/grf_bimat_campaign.yaml` showcases the materials library + 5×3 layout.
- New tests: `tests/test_campaign.py` — slot introspection, ref resolution (scalar/list/literal-override/missing-material/wrong-arity), YAML+JSON loading, `_select_pairs` (full grid / limit / shuffle reproducibility), end-to-end limit run with stub backend, dry-run side-effect-free check. **22 new tests → 95 total.**

### 2026-04-24
- Added `make_grf_bimat(h, w, correlation_length, volume_fraction, seed, ...)` — pure-numpy FFT-based Gaussian random field thresholded to two phases (TPU / PLA defaults in MPa-mm-ms units).
- Registered `grf_bimat` in `campaign._GEO_GENERATORS`; added `backend` field on `CampaignConfig` so campaigns can pick `freefem` or `jaxfem` per run.
- Authored `notebooks/grf_bimat_demo.ipynb` (13 cells: single sample → sweep → load case → solve → slice campaign → CLI handoff); added `notebook` extra (`jupyter`, `ipykernel`, `matplotlib`).
- Added `configs/grf_bimat_campaign.json` (10 geometries × 3 load cases = 30 samples). First full run: 30/30 after switching the JAX-FEM solver to UMFPACK (direct LU; the default bicgstab fails on high-E-contrast point loads).
- New tests: `test_geometry.py` GRF triad (shape, determinism, volume-fraction tolerance) → 73 total.
- Refreshed `README.md` and `CLAUDE.md` so the JAX-FEM install recipe, `petsc4py` gotcha, and new backend/campaign surface are documented.

### 2026-04-23
- Added `jaxfem` backend (`backends/jaxfem.py`): orchestration mode (loads user Python `solve(config)`), registered via `@register("jaxfem")`, lazy import-skip when missing.
- Added in-package JAX-FEM 2D elasticity solver (`solvers/elasticity2d_jaxfem.py`) mirroring the FreeFEM `.edp` I/O contract (same `geometry.dat`/`boundary.dat` input, same `fields_step_N.tsv` output). Per-pixel (E, ν) via `internal_vars`; BCs via penalty method in `get_mass_map`.
- Threaded `backend="freefem"|"jaxfem"` kwarg through `pixel_to_fem.run_simulation`.
- Smoke test + roundtrip + physics tests: `test_jaxfem.py`, `test_jaxfem_smoke.py`, `test_fem_roundtrip_jaxfem.py` (all 8 physics assertions match FreeFEM qualitatively).
- Fixture `tests/fixtures/jaxfem_elasticity_smoke.py` runnable both through the backend and as `python path/to/fixture.py --run-dir OUT`.

### 2026-03-20
- Consolidated `fem_framework` + `fem_dataset` into `fem-sim`.
- Unified CLI: `fem-sim run|index|video-manifest|build-dataset|inspect`.
- Consolidated duplicate FreeFEM binary detection into `freefem_binary.py`.

### 2026-03-19
- Extracted FEM dataset pipeline from `fem_world_model` into standalone package.

## Commands

```bash
# Base install
uv sync

# Optional extras
uv sync --extra jaxfem      # JAX + jax-fem (+ transitive meshio/gmsh/basix/pyfiglet)
uv sync --extra notebook    # jupyter + ipykernel + matplotlib
uv sync --extra viz         # matplotlib only (for `fem-sim inspect`)

# petsc4py is required by the jaxfem backend but must be installed manually
# (system PETSc needed; uv sync cannot build it without PETSC_DIR):
brew install petsc                                               # macOS
export PETSC_DIR=$(brew --prefix petsc) OMPI_CC=/usr/bin/clang
uv pip install 'petsc4py==3.24.*'                                # version must match brew's petsc
# NOTE: `uv sync` removes petsc4py each time; re-run this install after any sync.

# --- Tests ---

uv run python -m unittest discover -s tests                      # 120 total (1 auto-skip)
uv run python -m unittest tests.test_geometry tests.test_load_case tests.test_config tests.test_campaign tests.test_runner
uv run python -m unittest tests.test_freefem tests.test_fem_roundtrip            # needs FreeFem++
uv run python -m unittest tests.test_jaxfem_smoke tests.test_fem_roundtrip_jaxfem # needs jax-fem + petsc4py

# --- CLI ---

# Orchestration-style single or batch run (FreeFEM, FEniCSx-stub, or jaxfem)
uv run fem-sim run script.edp [KEY=VALUE ...]
uv run fem-sim run --params configs/framework_smoke.json

# Dataset campaign (geometry × load_case → solver → .npz). JSON or YAML.
uv run fem-sim build-dataset --config configs/elasticity2d_campaign.json
uv run fem-sim build-dataset --config configs/grf_bimat_campaign.yaml      # TPU × PLA via jaxfem (15 samples)
uv run fem-sim build-dataset --config configs/bimat_characterization_campaign.yaml  # 3 GRF × 8 characterization modes (24 samples)
# Subset before committing to the full grid:
uv run fem-sim build-dataset --config configs/grf_bimat_campaign.yaml --limit 4 --dry-run
uv run fem-sim build-dataset --config configs/grf_bimat_campaign.yaml --limit 4
uv run fem-sim build-dataset --config configs/grf_bimat_campaign.yaml --limit 4 --shuffle --seed 7
# Inline VTK export (writes vtk/<sample_id>/ alongside samples/<sample_id>.npz):
uv run fem-sim build-dataset --config configs/grf_bimat_campaign.yaml --export-vtk

# Inspect a .npz sample
uv run fem-sim inspect sample.npz --step 5

# Export a sample as a time-series VTK collection (open the .pvd in ParaView)
uv run fem-sim export-vtk sample.npz

# Index / video manifest
uv run fem-sim index outputs/runs --output outputs/datasets/index.json
uv run fem-sim video-manifest outputs/runs/smoke

# Demo notebook
uv run jupyter lab notebooks/grf_bimat_demo.ipynb
```

## Architecture

The package has two main capabilities under a unified namespace:

### Simulation Orchestration

Multi-backend FEM execution through pluggable backends.

- **`config.py`** — `SimulationConfig`, `load_config`, `load_batch`
- **`result.py`** — `RunResult` dataclass
- **`backends/__init__.py`** — Lazy registry with `@register` decorator
- **`backends/base.py`** — `BackendRunner` protocol
- **`backends/freefem.py`** — FreeFEM subprocess backend
- **`backends/fenicsx.py`** — FEniCSx stub (requires dolfinx)
- **`backends/jaxfem.py`** — JAX-FEM in-process backend; runs user `solve(config)` Python modules
- **`dataset.py`** — `build_dataset_index()`
- **`video.py`** — `build_video_manifest()`

### Dataset Generation

Pixel-geometry-driven FEM dataset generation.

- **`geometry.py`** — 6 generators → `(5, H, W)` arrays (solid_mask, material_id, E, nu, rho). Includes `make_grf_bimat` for random two-phase microstructures (TPU × PLA defaults).
- **`load_case.py`** — 5 BC generators → `(6, H, W)` arrays (disp_mask, force_mask, dx, dy, fx, fy)
- **`pixel_to_fem.py`** — `run_simulation(geo, bc, ..., backend="freefem"|"jaxfem")` — backend-agnostic bridge + `FEMSample` dataclass + `.npz` save/load
- **`campaign.py`** — Campaign orchestrator: geometry × load_case → FEM → `.npz`. `CampaignConfig` carries `backend`, an optional named `materials` library, and a list of geometry/load_case specs. `from_file()` loads JSON or YAML. `build_campaign(config, limit=None, shuffle=False, seed=None, dry_run=False)` supports running a subset before the full sweep.  `build_dataset(path_or_config, *, output_dir, steps, backend, materials, limit, shuffle, seed, dry_run)` is the notebook-friendly wrapper that loads from a path and applies kwarg overrides via `dataclasses.replace` (without mutating the loaded config). Sample IDs are `g{gi:03d}_l{li:02d}_<labels>` so they're stable across partial and full runs. Materials are resolved via signature introspection: any generator with `E*/nu*/rho*` parameter groups (slot suffixes `""`, `"_A"`, `"1"`, …) auto-supports the `materials:` field — scalar for single-slot, list for multi-slot.
- **`runner.py`** — `run_fem(script, *, params, backend, binary, plot, extra_params, **kv) -> list[RunResult]`. Notebook-friendly wrapper for `fem-sim run`; either `script` (single) or `params` (batch JSON/YAML). `**kv` becomes flat `-KEY VALUE` args. Validation failures raise `ValueError`.
- **`inspect.py`** — Matplotlib visualization of `.npz` samples
- **`solvers/elasticity2d.edp`** — FreeFEM plane-stress solver
- **`solvers/elasticity2d_jaxfem.py`** — JAX-FEM solver with matching I/O contract (same `.dat` / `.tsv` files)

### Shared

- **`freefem_binary.py`** — Consolidated FreeFEM binary detection (PATH + macOS app bundle)
- **`cli.py`** — Unified CLI with all subcommands

### Data Flow

```
geometry.py → load_case.py → pixel_to_fem.py → .npz samples
                                ↑ dispatches to FreeFEM (.edp)
                                    or JAX-FEM (elasticity2d_jaxfem.py)
campaign.py orchestrates the above for N geometries × M load cases
```

Both solvers read/write the same `geometry.dat`, `boundary.dat`,
`fields_step_N.tsv` files inside `run_dir`, so downstream loaders
(`load_results`, `save_sample`) are backend-agnostic.

### Dataset Format Contract

Each `.npz` contains: `geometry (5,H,W)`, `boundary (6,H,W)`, `fields (T,5,H,W)`.
Void pixels = 0.0. Variable `H, W`. This is the sole coupling point to `fem-world-model`.

## Key Design Points

- **Geometry-first**: The pixel image `(C_geo, H, W)` is the source of truth. FEM mesh is derived from it, results mapped back to same grid.
- **Lightweight core**: numpy only; jax/jax-fem are optional (`--extra jaxfem`), matplotlib optional (`--extra viz`), jupyter optional (`--extra notebook`).
- **Backend registry**: `@register("freefem"|"jaxfem"|...)` decorator, lazy-loaded. New backends implement `BackendRunner` protocol.
- **JAX-FEM solver**: Uses UMFPACK (direct sparse LU) for the pixel-grid elasticity problem because bicgstab diverges on high-E-contrast patterns (TPU/PLA) and point loads. Slower for very large grids; fine for ≤ 100×100 today.
- **FreeFEM binary detection**: Unified in `freefem_binary.py` — checks PATH for FreeFem++/freefem++/ff-mpirun, then macOS app bundle.

## Coding Conventions

- 4-space indentation
- `snake_case` for functions/files/variables, `PascalCase` for classes
- Test files named `test_<unit>.py` under `tests/`
- No linter configured yet; prefer `ruff` when adding one

## Known Issues & TODOs

- FEniCSx backend is a stub (not implemented).
- Campaign runner is single-threaded (could benefit from parallel execution; easy win since JAX-FEM solves are ~1–2 s on the 48×96 grid).  Use `--limit` for now to keep iteration fast.
- `petsc4py` cannot be declared as a dep in `pyproject.toml` — its pip build needs `PETSC_DIR` at install time. It gets wiped on every `uv sync`; document and require the manual reinstall.  Use `uv run --no-sync ...` for testing to avoid re-wiping.
- `sim/freefem/examples/` contains many tutorial scripts — consider trimming to essentials.
- Two parallel "jaxfem" code paths: `backends/jaxfem.py` is the *orchestration* backend (loads a user `.py` exposing `solve(config)`), while `pixel_to_fem._run_jaxfem` calls the in-package `solvers/elasticity2d_jaxfem.solve(nx, ny, steps, run_dir)` directly.  Two different `solve()` signatures — worth unifying.
- `backends/jaxfem.py` uses `os.chdir` for working-directory handoff, which blocks any future parallel execution.  Replace with `contextlib.chdir` (3.11+) or pass the dir as a parameter to `solve(config)`.
