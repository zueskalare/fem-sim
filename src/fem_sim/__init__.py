"""fem_sim — FEM simulation orchestration and dataset generation for ML training.

Combines two capabilities:
  1. Multi-backend FEM simulation orchestration (FreeFEM, FEniCSx, etc.)
  2. Pixel-geometry-driven dataset generation:
     geometry (5, H, W) + boundary conditions (6, H, W)
     -> FreeFEM solve -> fields (T, 5, H, W)
     -> .npz samples ready for ML training.
"""

# --- Dataset generation ---
from fem_sim.geometry import (
    N_GEO_CHANNELS,
    CH_SOLID,
    CH_MATID,
    CH_E,
    CH_NU,
    CH_RHO,
    make_rectangle,
    make_plate_with_hole,
    make_lshape,
    make_porous,
    make_bimat_rectangle,
    make_grf_bimat,
    save_geometry,
    load_geometry,
)
from fem_sim.load_case import (
    N_BC_CHANNELS,
    CH_DISP_MASK,
    CH_FORCE_MASK,
    CH_DX,
    CH_DY,
    CH_FX,
    CH_FY,
    make_cantilever_point_load,
    make_cantilever_distributed,
    make_simply_supported_distributed,
    make_displacement_bc,
    make_top_load_fixed_bottom,
    make_uniaxial,
    make_shear,
    make_biaxial,
    save_load_case,
    load_load_case,
)
from fem_sim.pixel_to_fem import (
    FEMSample,
    N_FIELD_CHANNELS,
    run_simulation,
    save_sample,
    load_sample,
)
from fem_sim.campaign import CampaignConfig, build_campaign, build_dataset

# --- Plugins ---
from fem_sim.plugins import get_plugin, list_plugins, register_plugin

# --- Simulation orchestration ---
from fem_sim.config import SimulationConfig, load_config, load_batch
from fem_sim.result import RunResult
from fem_sim.backends import get_backend, list_backends
from fem_sim.dataset import build_dataset_index
from fem_sim.runner import run_fem
from fem_sim.video import build_video_manifest

# --- Visualization ---
from fem_sim.inspect import inspect
from fem_sim.vtk_export import export_sample_vtk

__all__ = [
    # Geometry
    "N_GEO_CHANNELS", "CH_SOLID", "CH_MATID", "CH_E", "CH_NU", "CH_RHO",
    "make_rectangle", "make_plate_with_hole", "make_lshape",
    "make_porous", "make_bimat_rectangle", "make_grf_bimat",
    "save_geometry", "load_geometry",
    # Load cases
    "N_BC_CHANNELS", "CH_DISP_MASK", "CH_FORCE_MASK",
    "CH_DX", "CH_DY", "CH_FX", "CH_FY",
    "make_cantilever_point_load", "make_cantilever_distributed",
    "make_simply_supported_distributed", "make_displacement_bc",
    "make_top_load_fixed_bottom",
    "make_uniaxial", "make_shear", "make_biaxial",
    "save_load_case", "load_load_case",
    # FEM bridge
    "FEMSample", "N_FIELD_CHANNELS",
    "run_simulation", "save_sample", "load_sample",
    # Campaign
    "CampaignConfig", "build_campaign", "build_dataset",
    # Plugins
    "get_plugin", "list_plugins", "register_plugin",
    # Simulation orchestration
    "SimulationConfig", "RunResult",
    "load_config", "load_batch",
    "get_backend", "list_backends",
    "run_fem",
    "build_dataset_index", "build_video_manifest",
    # Visualization
    "inspect", "export_sample_vtk",
]
