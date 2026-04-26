"""Visualize an FEM dataset sample (.npz) using matplotlib.

Usage:
    fem-sim inspect outputs/datasets/sample_0000.npz
    fem-sim inspect outputs/datasets/sample_0000.npz --step 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def inspect(npz_path: str | Path, step: int | None = None) -> None:
    """Plot geometry, BCs, and field channels from a .npz sample."""
    import matplotlib.pyplot as plt

    sample = np.load(npz_path)
    geo = sample["geometry"]     # (C_geo, H, W)
    bc = sample["boundary"]      # (C_bc, H, W)
    fields = sample["fields"]    # (T, C_field, H, W)

    n_steps = fields.shape[0]
    if step is None:
        step = n_steps - 1  # show final load step by default
    step = min(step, n_steps - 1)

    fig, axes = plt.subplots(3, 6, figsize=(20, 10))

    geo_names = ["solid_mask", "material_id", "E", "nu", "rho"]
    bc_names = ["disp_mask", "force_mask", "dx", "dy", "fx", "fy"]
    field_names = ["ux", "uy", "\u03c3xx", "\u03c3yy", "\u03c3xy"]
    field = fields[step]

    rows = (
        ("Geo", geo, geo_names, None),
        ("BC", bc, bc_names, None),
        (f"Field[{step}]", field, field_names, "RdBu_r"),
    )
    for r, (label, data, names, cmap) in enumerate(rows):
        for c in range(6):
            ax = axes[r, c]
            if c < len(names):
                im = ax.imshow(data[c], origin="lower", aspect="auto", cmap=cmap)
                ax.set_title(f"{label}: {names[c]}")
                plt.colorbar(im, ax=ax, shrink=0.7)
            else:
                ax.axis("off")

    fig.suptitle(f"Sample: {Path(npz_path).stem}  |  Step {step}/{n_steps - 1}", fontsize=14)
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect FEM dataset sample")
    parser.add_argument("npz_path", help="Path to .npz sample file")
    parser.add_argument("--step", type=int, default=None,
                        help="Load step to visualize (default: last)")
    args = parser.parse_args()
    inspect(args.npz_path, args.step)


if __name__ == "__main__":
    main()
