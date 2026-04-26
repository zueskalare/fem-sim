"""Dataset index builder.

Scans a directory tree for ``.npz`` samples produced by ``build_campaign``
and writes a JSON index of every sample.  Each entry records the path to
the ``.npz``, its sidecar metadata file (if present), and metadata fields
copied from the sidecar (sample_id, geometry_spec, load_case_spec, etc.).

The directory layout produced by ``fem_sim.campaign.build_campaign`` is::

    <campaign_root>/
        runs/<sample_id>/                 (raw FEM run outputs)
        samples/<sample_id>.npz           (the canonical sample)
        samples/<sample_id>.json          (sidecar metadata)
        index.json                        (campaign-level index)

``build_dataset_index`` is the cross-campaign aggregator: point it at any
parent directory and it discovers every ``.npz`` recursively.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_dataset_index(runs_root: Path, output_path: Path) -> Path:
    """Scan ``runs_root`` recursively for ``.npz`` samples and write a JSON index.

    Parameters
    ----------
    runs_root : Path
        Either a campaign root (with ``samples/*.npz``) or any parent directory
        containing one or more campaigns.  Search is recursive.
    output_path : Path
        Destination for the index JSON.

    Returns
    -------
    Path
        ``output_path`` (with parents created).
    """
    entries: list[dict[str, object]] = []
    for npz_path in sorted(runs_root.rglob("*.npz")):
        meta_path = npz_path.with_suffix(".json")
        metadata = _load_metadata(meta_path)
        entries.append({
            "sample_id": metadata.get("sample_id", npz_path.stem),
            "backend": metadata.get("backend", "unknown"),
            "npz": str(npz_path),
            "metadata": str(meta_path) if meta_path.exists() else None,
            "run_dir": metadata.get("run_dir"),
            "nx": metadata.get("nx"),
            "ny": metadata.get("ny"),
            "steps": metadata.get("steps"),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"samples": entries}, indent=2),
        encoding="utf-8",
    )
    return output_path


def _load_metadata(meta_path: Path) -> dict[str, object]:
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))
