# fem-sim

FEM simulation orchestration and dataset generation for ML training.

Combines two capabilities in one lightweight package:

1. **Simulation orchestration** — run FEM solvers (FreeFEM, FEniCSx, JAX-FEM) through a unified CLI and Python API.
2. **Dataset generation** — produce spatially aligned 2D FEM datasets from pixel geometry images, driven by JSON or YAML campaign configs with a named materials library and `(geometry × load_case)` sweeps.

The pixel image is the source of truth — FEM mesh is derived from it, results mapped
back to the same grid. All tensors (geometry, BCs, fields) are aligned at the same `H x W`.

Every CLI subcommand has a matching Python function (`build_dataset`,
`run_fem`, `inspect`, `export_sample_vtk`) so the same code drives the CLI
and the notebook. Optional `--export-vtk` produces a ParaView-ready
`.vti` + `.pvd` time-series alongside each `.npz` sample.

## Quick Start

```bash
# Install
uv add fem-sim

# Run a FEM simulation
fem-sim run script.edp mesh_nx=32 solver_dt=0.01

# Preview the campaign without solving anything
fem-sim build-dataset --config configs/grf_bimat_campaign.yaml --limit 4 --dry-run

# Build a small subset first (sanity check before the full sweep)
fem-sim build-dataset --config configs/grf_bimat_campaign.yaml --limit 4

# Build the full dataset
fem-sim build-dataset --config configs/grf_bimat_campaign.yaml

# Same, but also write a ParaView animation per sample
fem-sim build-dataset --config configs/grf_bimat_campaign.yaml --export-vtk

# Inspect a sample (requires matplotlib)
pip install fem-sim[viz]
fem-sim inspect outputs/datasets/grf_bimat_v2/samples/g000_l00_grf_bimat_48x96_cantilever_distributed.npz
```

## Python API

### One sample, end-to-end

```python
from fem_sim import make_rectangle, make_cantilever_distributed
from fem_sim import run_simulation, save_sample

geo = make_rectangle(h=16, w=32, E=210_000, nu=0.3, rho=7800)
bc = make_cantilever_distributed(geo, load_mag=-500, direction="y")
sample = run_simulation(geo, bc, steps=10, run_dir="outputs/my_run", backend="freefem")
save_sample(sample, "outputs/my_sample.npz")

print(sample.fields.shape)  # (10, 5, 16, 32) — T steps, 5 field channels, H, W
```

| Parameter   | Meaning                                                                                                                       |
|-------------|-------------------------------------------------------------------------------------------------------------------------------|
| `h, w`      | Pixel grid height / width. The FEM mesh has one quad per pixel; output is mapped back to this grid.                           |
| `E, nu, rho`| Young's modulus, Poisson's ratio, density. Units are user-defined (the example uses MPa-mm-tonne; the GRF demo uses MPa-mm-ms).|
| `load_mag`  | **Total** load magnitude. For distributed loads it's spread evenly across the loaded edge; for point loads it lives at one pixel. Negative = downward when `direction="y"`. |
| `direction` | `"y"` (default) or `"x"`. Controls which force/displacement channel is populated.                                              |
| `steps`     | Number of quasi-static load increments. Step `t` applies load `(t+1)/steps` of the full magnitude. Output shape's first axis is `T = steps`. |
| `backend`   | `"freefem"` (subprocess) or `"jaxfem"` (in-process). Both produce identical `.npz` payloads.                                  |

### Notebook-friendly campaign runner

Every CLI subcommand has a matching Python function so the same code can drive
the CLI and the notebook:

```python
from fem_sim import build_dataset, inspect

# Equivalent to:
#   fem-sim build-dataset --config configs/grf_bimat_campaign.yaml --limit 4
paths = build_dataset("configs/grf_bimat_campaign.yaml", limit=4)

# Override fields without editing the YAML
paths = build_dataset(
    "configs/grf_bimat_campaign.yaml",
    output_dir="outputs/notebook_run",   # where samples/runs/index.json land
    steps=3,                             # shorter ramp than the YAML specifies
    limit=2,                             # only run the first 2 (geo, lc) pairs
)

# Visualize a sample (matplotlib required)
inspect(paths[0], step=2)                # which load-step to plot (default: last)

# Export the same sample as VTK for ParaView (no extra deps)
from fem_sim import export_sample_vtk
pvd = export_sample_vtk(paths[0])        # writes <stem>_vtk/step_*.vti + .pvd
```

`build_dataset` accepts a path or an already-loaded `CampaignConfig`. Every
keyword argument below is optional and defaults to "leave the loaded config
alone":

| Kwarg          | Type                            | Meaning                                                                                                                                                           |
|----------------|---------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `output_dir`   | path                            | Override where `runs/`, `samples/`, `vtk/` (when `export_vtk=True`), and `index.json` are written.                                                                  |
| `steps`        | int                             | Override the number of quasi-static load increments per sample.                                                                                                    |
| `backend`      | `"freefem"` / `"jaxfem"`        | Override which solver runs every sample.                                                                                                                          |
| `materials`    | `dict[name, {E, nu, rho}]`      | Override the materials library (handy for sweeping properties from Python).                                                                                       |
| `limit`        | int                             | Run only the first **N** `(geometry, load_case)` pairs.                                                                                                            |
| `shuffle`      | bool                            | Randomize pair order before applying `limit`.                                                                                                                     |
| `seed`         | int                             | RNG seed for `shuffle`. Same seed → same selection.                                                                                                                |
| `dry_run`      | bool                            | Log the planned sample IDs and return `[]` without solving or creating any output directories.                                                                    |
| `export_vtk`   | bool                            | After saving each `.npz`, also write a time-series VTK collection under `<output_dir>/vtk/<sample_id>/` for ParaView animation.                                    |

Overrides use `dataclasses.replace` so the loaded config object is *not*
mutated — you can pass the same `CampaignConfig` to multiple `build_dataset`
calls with different overrides.

### Simulation Orchestration

```python
from fem_sim import run_fem

# Single FEM run; kwargs become -KEY VALUE script args
results = run_fem("script.edp", mesh_nx=32, dt=0.01)
result = results[0]
print(result.succeeded, result.outputs)

# Batch sweep — point at a JSON or YAML batch file
results = run_fem(params="configs/sweep.json", backend="jaxfem")
```

| Kwarg            | Meaning                                                                                                                                  |
|------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `script`         | Path to a single solver script. Mutually exclusive with `params`.                                                                        |
| `params`         | Path to a batch JSON/YAML file with a top-level `runs:` list. Mutually exclusive with `script`.                                          |
| `backend`        | Force a backend on every config (overrides what's in the file).                                                                          |
| `binary`         | Override the solver binary path on every config.                                                                                         |
| `plot`           | Set `backend_options["plot"]=True` (FreeFEM keeps its GUI window open).                                                                   |
| `extra_params`   | A dict of `-KEY VALUE` pairs merged into every config's `params`.                                                                        |
| `**kv`           | Convenience: keyword args that become `-KEY VALUE`. Take precedence over `extra_params` on key conflicts.                                |

Validation failures (missing binary, missing script, etc.) raise `ValueError`
so they're easy to catch in notebooks. For finer control the lower-level
primitives are still exported (`SimulationConfig`, `load_config`, `load_batch`,
`get_backend`, `list_backends`).

## CLI

`fem-sim` is the unified entry point. All subcommands accept `-h`/`--help`.

| Subcommand        | Purpose                                                                     | Python equivalent          |
|-------------------|-----------------------------------------------------------------------------|----------------------------|
| `run`             | Run one or many FEM scripts (FreeFEM `.edp` or jaxfem `.py`).               | `run_fem(...)`             |
| `build-dataset`   | Run a `(geometry × load_case)` campaign from a JSON/YAML config.            | `build_dataset(...)`       |
| `inspect`         | Plot a `.npz` sample (geometry, BCs, fields) with matplotlib.               | `inspect(path, step)`      |
| `export-vtk`      | Write a time-series `.vti` + `.pvd` collection for one `.npz`.              | `export_sample_vtk(path)`  |
| `index`           | Recursively scan a tree for `.npz` samples and write a JSON index.          | `build_dataset_index(...)` |
| `video-manifest`  | List a single run's `fields_step_*.tsv` files in step order.                | `build_video_manifest(...)`|

### `fem-sim run` — single or batch FEM solver invocation

```bash
fem-sim run script.edp [KEY=VALUE ...]
fem-sim run --params batch.json [KEY=VALUE ...]
```

| Argument           | Type    | Meaning                                                                                                  |
|--------------------|---------|----------------------------------------------------------------------------------------------------------|
| `script` (positional) | path    | Solver script: `.edp` for FreeFEM, `.py` exposing `solve(config)` for jaxfem.                           |
| `--params PATH`    | path    | JSON / YAML batch file with a top-level `runs:` list — runs every entry in turn.                         |
| `--backend NAME`   | str     | Override the backend on every loaded config: `freefem`, `jaxfem`, or `fenicsx` (stub).                   |
| `--binary PATH`    | path    | Override the solver binary (e.g. point at a specific `FreeFem++` build).                                  |
| `--plot`           | flag    | Keep the FreeFEM GUI window open (drops the default `-nw` flag). FreeFEM only.                            |
| `KEY=VALUE`        | varargs | Passed verbatim as `-KEY VALUE` to the solver script. Strings auto-cast to int/float/bool when they parse. |

### `fem-sim build-dataset` — dataset campaign

```bash
fem-sim build-dataset --config campaign.yaml [SAMPLING FLAGS]
```

| Flag                | Default      | Meaning                                                                                                                              |
|---------------------|--------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `--config PATH`     | required     | JSON or YAML campaign config (auto-detected by extension). See the [Dataset Campaign](#dataset-campaign) section for the schema.     |
| `--limit N`         | full grid    | Run only the first **N** `(geometry, load_case)` pairs out of the full `n_geo × n_lc` grid. Use this for sanity-check runs.            |
| `--shuffle`         | off          | Randomize pair order *before* applying `--limit`. Combine with `--seed` for reproducibility.                                          |
| `--seed N`          | fresh RNG    | RNG seed for `--shuffle`. Same seed → same pair selection across machines and runs.                                                   |
| `--dry-run`         | off          | Log the planned sample IDs and exit without solving anything (no output dirs created). Use this to preview the plan.                  |
| `--export-vtk`      | off          | After saving each `.npz`, also export a time-series VTK collection (`.vti` per step + `.pvd`) under `<output_dir>/vtk/<sample_id>/`. Open the `.pvd` in ParaView. |

Sample IDs are `g{gi:03d}_l{li:02d}_<labels>` and stable on `(gi, li)`, so a
`--limit 4` run and a later full run produce identical filenames for any
overlapping pair — sample subsets accumulate cleanly in the same output dir.

### Other subcommands

```bash
# Visualize a .npz sample (matplotlib required)
fem-sim inspect sample.npz [--step N]      # default: last load step

# Export a sample as a time-series VTK collection (open the .pvd in ParaView)
fem-sim export-vtk sample.npz [--output DIR]

# Aggregate every .npz under runs_root into one JSON index
fem-sim index outputs/runs --output outputs/datasets/index.json

# Per-run manifest of fields_step_*.tsv (sorted by integer step)
fem-sim video-manifest outputs/runs/smoke [--output PATH]
```

| Argument               | Meaning                                                                                                                  |
|------------------------|--------------------------------------------------------------------------------------------------------------------------|
| `inspect --step N`     | Which load step to plot in the field channels row. Defaults to the final step (`T-1`).                                   |
| `export-vtk --output`  | Output directory for `step_NNNN.vti` files + a `.pvd` collection. Default: `<sample_dir>/<stem>_vtk/` next to the `.npz`. |
| `index runs_root`      | Recursively scans `runs_root` for `*.npz` samples and writes a JSON index pointing at every `.npz` + sidecar JSON.        |
| `video-manifest`       | Lists the `fields_step_*.tsv` files in a single run dir, sorted by integer step (lex sort breaks at ≥ 10 steps).           |

### VTK export details

`export-vtk` produces:

- One `step_NNNN.vti` (VTK ImageData) per load step, with cell-centred
  arrays for every channel: `solid_mask`, `material_id`, `E`, `nu`, `rho`,
  `disp_mask`, `force_mask`, `prescribed_disp` (vector), `prescribed_force`
  (vector), `displacement` (vector — the time-varying ux/uy at this step),
  `stress_xx`, `stress_yy`, `stress_xy`.
- One `<sample_stem>.pvd` collection that ties the steps together with
  `timestep` set to the load fraction (`(t+1)/T` ∈ [0, 1]).

Open the `.pvd` in ParaView to play the load-ramp animation. The pixel grid
is exported as ImageData (uniform spacing 1 in each axis), which is the
most compact VTK representation for regular grids — no mesh / connectivity
needed.

## Tensor Format

| Tensor   | Shape          | Channels                                        | Notes                                                                                |
|----------|----------------|-------------------------------------------------|--------------------------------------------------------------------------------------|
| Geometry | `(5, H, W)`    | `solid_mask, material_id, E, nu, rho`           | Material properties per pixel. Void pixels = 0.0 in every channel.                   |
| Boundary | `(6, H, W)`    | `disp_mask, force_mask, dx, dy, fx, fy`         | Mask channels are `1.0` where a BC is applied; value channels carry the prescribed `u` or `f`. |
| Fields   | `(T, 5, H, W)` | `ux, uy, sxx, syy, sxy`                         | One tensor per load step. **`T = steps`**: the load is ramped linearly from 0 → full magnitude over `T` quasi-static increments; channels are displacement (mm) and stress (MPa) per pixel. |

`H, W` vary across samples (the geometry generator decides). Within one
sample, all three tensors share the same `H, W`. The first axis of `Fields`
is time-ordered: `fields[0]` is the first increment (`load = 1/T`), `fields[-1]`
is the fully loaded state.

Channel index constants are exported (`CH_SOLID`, `CH_E`, `CH_DX`, `CH_FY`, …)
so downstream code doesn't need magic numbers.

## Available Geometries

All generators take `h, w` (pixel grid size) plus material properties (`E, nu, rho`,
or `E1/nu1/rho1` + `E2/nu2/rho2` for bi-material variants). When invoked from a
campaign config you can replace the literal material kwargs with a `materials:`
reference — see [Dataset Campaign](#dataset-campaign).

| Generator                  | Shape                                       | Geometry-specific params                                                                                       |
|----------------------------|---------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| `make_rectangle`           | Solid block, single material                | (none)                                                                                                          |
| `make_plate_with_hole`     | Plate with a circular void                  | `cx, cy` (hole centre, normalized [0,1]); `r` (radius, fraction of `min(h,w)/max(h,w)`)                         |
| `make_lshape`              | Rectangle with top-right quadrant cut       | `cut_frac` (fraction of each dim removed, default 0.5)                                                          |
| `make_porous`              | Plate with random circular pores            | `n_pores`, `pore_r_range=(r_min, r_max)`, `seed`, `margin` (keep pore centres ≥ margin from boundary)            |
| `make_bimat_rectangle`     | Two materials split vertically              | `split_frac` (left material's column fraction); `E1/nu1/rho1` + `E2/nu2/rho2`                                   |
| `make_grf_bimat`           | Gaussian-random-field two-phase composite   | `correlation_length` (feature size in pixels — larger = blobbier), `volume_fraction` (target fraction of phase A), `seed`; `E_A/nu_A/rho_A` + `E_B/nu_B/rho_B` (TPU × PLA defaults) |
| `make_spinodoid` *(planned — see [docs/spinodoid.md](docs/spinodoid.md))* | Anisotropic plane-wave-sum bi-material microstructure (Kumar et al. 2020, 2D) | `theta1_deg, theta2_deg` (cone half-angles around `±x` / `±y` controlling anisotropy: isotropic / lamellar / cubic / oblique), `wavelength` (characteristic feature size in px), `volume_fraction`, `n_waves`, `seed`; `E_A/nu_A/rho_A` + `E_B/nu_B/rho_B` |

## Design docs

- **[docs/spinodoid.md](docs/spinodoid.md)** — 2D spinodoid bi-material
  generator: math (eq. 1–3), admissible-direction set, algorithm,
  morphology gallery, and the planned `make_spinodoid` API. Adapted from
  Kumar et al. *npj Comput Mater* **6**, 73 (2020).

## Available Load Cases

All load cases take a `geo` array first, then their own params. `direction` is
`"x"` or `"y"` (the axis the prescribed force/displacement points along).
`load_mag` is **total** for distributed loads (split across the loaded edge)
and **per-pixel** for point loads. Engineering sign convention: positive
displacement → tension, negative → compression.

### Bending / loading suite

| Generator                            | Boundary conditions                                      | Load params                              |
|--------------------------------------|----------------------------------------------------------|------------------------------------------|
| `make_cantilever_point_load`         | Left edge fixed; point force on right edge               | `load_mag`, `load_pos` (∈[0,1]: 0=bottom of right edge, 1=top), `direction` |
| `make_cantilever_distributed`        | Left edge fixed; distributed load on right edge          | `load_mag`, `direction`                   |
| `make_simply_supported_distributed`  | Both bottom corners pinned; distributed load on top edge | `load_mag`, `direction`                   |
| `make_displacement_bc`               | Left edge fixed; prescribed displacement on right edge   | `disp_mag`, `direction`                   |
| `make_top_load_fixed_bottom`         | Bottom edge fixed; distributed load on top edge          | `load_mag`, `direction`                   |

### Characterization suite (square bimaterial samples)

These prescribe displacement on opposing edges so the central region
experiences a near-uniform stress state — the standard FEM textbook setup
for material characterization. Use any geometry; works best on square
samples (`h == w`). See `configs/bimat_characterization_campaign.yaml` for
a 3 × 8 example dataset.

| Generator        | Boundary conditions                                                                                                                | Load params                          |
|------------------|------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|
| `make_uniaxial`  | `direction='x'`: left clamped, right clamped at `(disp_mag, 0)`. `direction='y'`: bottom clamped, top clamped at `(0, disp_mag)`.   | `disp_mag`, `direction`              |
| `make_shear`     | Simple shear. `direction='x'` (horizontal shear): bottom clamped, top clamped at `(disp_mag, 0)`. `direction='y'` (vertical shear): left clamped, right clamped at `(0, disp_mag)`. | `disp_mag`, `direction`              |
| `make_biaxial`   | Bottom + left edges pinned; top edge gets `(0, disp_y)`; right edge gets `(disp_x, 0)`; corners blend naturally — top-right ends up at `(disp_x, disp_y)`. | `disp_x`, `disp_y` (signed independently) |

**Note**: clamped edges constrain both `ux` and `uy` (the BC tensor uses a
single per-pixel `disp_mask` channel — there is no per-component mask), so
loaded edges cannot Poisson-contract laterally and corners have stress
concentrations. The central region still gives clean characterization signal.

## Supported Backends

| Backend   | Status       | Solver          | Integration   |
|-----------|-------------|-----------------|---------------|
| `freefem` | Implemented | FreeFEM+ binary | Subprocess    |
| `fenicsx` | Stub        | FEniCSx/dolfinx | In-process    |
| `jaxfem`  | Implemented | JAX-FEM         | In-process    |

New backends can be added by implementing the `BackendRunner` protocol and registering with `@register`.

### JAX-FEM backend

The `jaxfem` backend has two use modes.

**Orchestration** — run a user Python script. The module must expose a
top-level `solve(config: SimulationConfig) -> dict` returning output paths:

```python
config = SimulationConfig(case_script="my_elasticity.py", backend="jaxfem")
result = get_backend("jaxfem").run(config)
```

**Dataset generation** — route the built-in pixel-grid 2D elasticity solver
through JAX-FEM (matches the FreeFEM `.edp` solver's I/O contract, so
downstream tooling is identical):

```python
sample = run_simulation(geo, bc, steps=10, run_dir="...", backend="jaxfem")
```

Install:

```bash
uv sync --extra jaxfem
# macOS: PETSc is an extra system dep required by jax-fem's solver
brew install petsc
uv pip install 'petsc4py==3.24.*'
```

## Notebook demo

`notebooks/grf_bimat_demo.ipynb` walks through the GRF bi-material pipeline
end-to-end: single-sample generation → 3×3 parameter sweep → load case →
JAX-FEM solve → visualization → batch campaign via the `build_dataset`
one-liner.

```bash
uv sync --extra notebook --extra jaxfem
# macOS: also install system PETSc (see JAX-FEM backend section)
uv run jupyter lab notebooks/grf_bimat_demo.ipynb
```

## Dataset Campaign

A campaign is a JSON or YAML file (auto-detected by extension) describing a
list of geometries and a list of load cases. The campaign produces one sample
per `(geometry, load_case)` pair — `n_geo × n_lc` samples for the full sweep.

```yaml
# Top-level fields
output_dir: outputs/datasets/my_campaign     # required: where samples/, runs/, index.json land
steps: 10                                    # quasi-static load increments per sample (default 10)
backend: jaxfem                              # which solver runs every sample (default "freefem")

# Optional named material library — referenced from geometry specs below
materials:
  TPU:   {E: 30.0,    nu: 0.48, rho: 1.2e-9}     # Young's modulus, Poisson's ratio, density
  PLA:   {E: 3500.0,  nu: 0.36, rho: 1.24e-9}    # units are user-defined; example uses MPa-mm-ms
  steel: {E: 210000,  nu: 0.30, rho: 7.8e-9}

# Each entry must have `type:` (matches a make_* generator name).
# The remaining keys map to the generator's kwargs.
geometries:
  - {type: rectangle, h: 32, w: 64,
     materials: steel}                                       # scalar form → single-material generator

  - {type: grf_bimat, h: 48, w: 96,
     correlation_length: 6.0,                                # GRF feature size in pixels
     volume_fraction: 0.5,                                   # target fraction of phase A
     seed: 0,                                                # reproducibility
     materials: [TPU, PLA]}                                  # list form → fills (E_A,nu_A,rho_A) and (E_B,nu_B,rho_B)

# Each entry must have `type:` (matches a make_* generator) plus its kwargs.
load_cases:
  - {type: cantilever_distributed, load_mag: -5.0}           # negative = downward
  - {type: cantilever_point_load,  load_mag: -20.0, load_pos: 1.0}
  - {type: top_load_fixed_bottom,  load_mag: -2.0}
```

### Top-level fields

| Field         | Required? | Default     | Meaning                                                                                              |
|---------------|-----------|-------------|------------------------------------------------------------------------------------------------------|
| `output_dir`  | yes       | —           | Root directory for the campaign. `runs/<sample_id>/`, `samples/<sample_id>.npz`, and `index.json` go here. |
| `steps`       | no        | `10`        | Quasi-static load increments per sample. Output `fields` shape is `(steps, 5, H, W)`.                |
| `backend`     | no        | `"freefem"` | `"freefem"` (subprocess) or `"jaxfem"` (in-process). Both produce identical `.npz` payloads.         |
| `freefem_binary` | no     | auto-detect | Override the FreeFEM binary path. Ignored for jaxfem.                                                |
| `materials`   | no        | `{}`        | Named library — see below.                                                                           |
| `geometries`  | yes       | —           | List of geometry specs (each with `type:` plus generator kwargs).                                    |
| `load_cases`  | yes       | —           | List of load case specs (each with `type:` plus generator kwargs).                                   |

### Materials library + slot resolution

Each geometry spec can pull material properties from the named library via a
`materials:` field:

- **Scalar form** (`materials: steel`) — for generators with a single material
  slot (e.g. `make_rectangle`, `make_porous`).
- **List form** (`materials: [TPU, PLA]`) — for multi-slot generators. List
  position determines which slot each material fills.

Slot inference is signature-based: any generator with `E*, nu*, rho*` parameter
groups (e.g. `E_A/nu_A/rho_A`, `E1/nu1/rho1`) automatically supports the
`materials` field — no registry to update. The slot suffix is the part after
`E` in the parameter name, in declaration order:

| Generator               | Slots          | List form maps to                                                                |
|-------------------------|----------------|----------------------------------------------------------------------------------|
| `make_rectangle`        | `[""]`         | `materials: steel` → `E=…, nu=…, rho=…`                                           |
| `make_bimat_rectangle`  | `["1", "2"]`   | `materials: [a, b]` → `E1=…, nu1=…, rho1=…, E2=…, nu2=…, rho2=…`                 |
| `make_grf_bimat`        | `["_A", "_B"]` | `materials: [a, b]` → `E_A=…, nu_A=…, rho_A=…, E_B=…, nu_B=…, rho_B=…`           |

Inline `E` / `nu` / `rho` keys still work and **override** the matching slot,
so you can pull most properties from the library and tweak just one:

```yaml
- {type: grf_bimat, h: 48, w: 96, materials: [TPU, PLA], E_A: 99.0}
  # → A is mostly TPU, but E_A is overridden to 99.0
```

### Subsetting a campaign

```bash
# Quick subset first (deterministic — first N pairs)
fem-sim build-dataset --config my_campaign.yaml --limit 4

# Random subset (reproducible with a seed)
fem-sim build-dataset --config my_campaign.yaml --limit 4 --shuffle --seed 7

# Preview the plan without solving anything
fem-sim build-dataset --config my_campaign.yaml --limit 4 --dry-run

# Full sweep
fem-sim build-dataset --config my_campaign.yaml
```

Sample IDs (`g{gi:03d}_l{li:02d}_<labels>`) are keyed on (geometry index,
load_case index) so a sampled run and a full run produce identical IDs for
overlapping pairs — sample subsets are extensions of the full grid, not
renumberings.

### Parallel campaigns with MPI (single node)

For million-sample dataset builds, `build_campaign` dispatches whole samples
across MPI ranks — each rank solves its own samples on its own process,
no inter-rank communication on the solve. Today the JAX-FEM backend is the
target (verified path); FreeFEM works trivially as a subprocess but isn't
yet wired into the dispatch.

```bash
# Install
uv sync --extra mpi             # adds mpi4py
brew install open-mpi            # macOS; or use your distro's openmpi-dev

# Run with N ranks
mpirun -n 8 uv run --no-sync fem-sim build-dataset \
    --config configs/grf_bimat_campaign.yaml
```

How it works:
- All ranks load the config and compute the same `pairs` list (deterministic).
- Rank `r` of size `N` takes the round-robin slice `pairs[r::N]`.
- Each rank writes its own samples under the shared `output_dir/samples/`;
  unique sample IDs prevent collisions.
- After a barrier, rank 0 collects all successful paths and writes a global
  `index.json` (with an `mpi_size` field for provenance).
- All ranks log their own per-sample work prefixed with `[rN/size]` so you can
  follow progress per rank.

The JAX-FEM solver pins `petsc4py` to `MPI.COMM_SELF` at import time so each
rank's PETSc objects are rank-local (otherwise jax-fem's `PETSc.Mat` defaults
to `MPI_COMM_WORLD` and assembly fails — every rank would think it's a slice
of one distributed matrix). Single-process runs are unaffected.

If `mpi4py` isn't installed, `build_campaign` silently runs single-process —
no flag needed.

### Bundled example configs

| File                                                | Layout              | Content                                                                                  |
|-----------------------------------------------------|---------------------|------------------------------------------------------------------------------------------|
| `configs/grf_bimat_campaign.yaml`                   | 5 × 3 = 15 samples  | TPU/PLA GRF microstructures × bending load cases. Sweeps `correlation_length`/`volume_fraction`. |
| `configs/bimat_characterization_campaign.yaml`      | 3 × 8 = 24 samples  | Square (64×64) GRF bimaterial × 8 characterization modes (uniaxial x±, uniaxial y±, shear x, shear y, biaxial tension, biaxial compression). |
| `configs/grf_bimat_campaign.json`                   | identical to YAML   | JSON variant of the GRF config — proves the loader is format-agnostic.                   |
| `configs/elasticity2d_campaign.json`                | small               | Minimal sanity check via FreeFEM.                                                        |
| `configs/framework_smoke.json`                      | smoke               | Single-run config used by the orchestration smoke test.                                  |

## Output Format

A campaign run produces, under `<output_dir>/`:

```
runs/<sample_id>/                   # raw FEM artifacts (geometry.dat, boundary.dat, fields_step_*.tsv, …)
samples/<sample_id>.npz             # the canonical sample (3 numpy arrays)
samples/<sample_id>.json            # sidecar metadata (sample_id, geometry_spec, load_case_spec, gi, li, backend, nx/ny/steps)
vtk/<sample_id>/                    # only present with --export-vtk: step_NNNN.vti + <sample_id>.pvd
index.json                          # campaign-level summary (total_samples, total_in_grid, limit, shuffle, seed, export_vtk, samples list)
```

Each `.npz` carries:

| Key        | Shape          | Meaning                                                    |
|------------|----------------|------------------------------------------------------------|
| `geometry` | `(5, H, W)`    | The geometry tensor described above.                       |
| `boundary` | `(6, H, W)`    | The boundary-condition tensor described above.             |
| `fields`   | `(T, 5, H, W)` | Time-ordered field snapshots, one per quasi-static step.   |

Sidecar metadata makes downstream loading easier — you can group / filter
samples by `sample_id`, `gi`, `li`, or `backend` without parsing filenames.

## Requirements

- Python >= 3.13
- numpy >= 2.2
- pyyaml >= 6.0
- FreeFem++ (optional — for the `freefem` backend and its tests)
- `jax-fem` + `jax` + `petsc4py` (optional — for the `jaxfem` backend;
  see the [JAX-FEM backend](#jax-fem-backend) section for the macOS
  install recipe)
- matplotlib >= 3.10 (optional — for `fem-sim inspect` and the demo notebook)
- jupyter + ipykernel (optional — for `notebooks/grf_bimat_demo.ipynb`)
- `mpi4py` >= 4.0 + an MPI implementation like Open MPI (optional — for
  `mpirun -n N fem-sim build-dataset` parallel campaigns; install via
  `uv sync --extra mpi` and your distro's `openmpi` / `brew install open-mpi`)

## Tests

```bash
# Pure-python subset (no FreeFEM, no JAX-FEM needed)
uv run python -m unittest tests.test_geometry tests.test_load_case \
    tests.test_config tests.test_plugins tests.test_jaxfem \
    tests.test_campaign tests.test_runner tests.test_vtk_export

# Full suite (124 tests; requires FreeFem++ for some, JAX-FEM stack for others)
uv run python -m unittest discover -s tests

# JAX-FEM backend only (needs petsc4py — see backend section)
uv run python -m unittest tests.test_jaxfem_smoke tests.test_fem_roundtrip_jaxfem
```
