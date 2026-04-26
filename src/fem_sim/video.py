"""Per-run video manifest builder.

Collects the time-ordered ``fields_step_*.tsv`` files written by either
the FreeFEM or JAX-FEM backend so a downstream renderer can iterate them
in load-step order.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_STEP_RE = re.compile(r"fields_step_(\d+)\.tsv$")


def build_video_manifest(run_dir: Path, output_path: Path) -> Path:
    """Write a JSON manifest of ``fields_step_N.tsv`` files in ``run_dir``.

    Files are sorted by their integer step index so the manifest reflects
    physical time order (lexicographic sort would break for ≥ 10 steps).
    """
    field_files = sorted(
        (p for p in run_dir.glob("fields_step_*.tsv") if _STEP_RE.search(p.name)),
        key=lambda p: int(_STEP_RE.search(p.name).group(1)),  # type: ignore[union-attr]
    )

    series_path = run_dir / "series.tsv"
    summary_path = run_dir / "summary.tsv"
    metadata_path = run_dir / "metadata.json"
    payload = {
        "run_dir": str(run_dir),
        "field_files": [str(p) for p in field_files],
        "n_steps": len(field_files),
        "series_file": str(series_path) if series_path.exists() else None,
        "summary_file": str(summary_path) if summary_path.exists() else None,
        "metadata_file": str(metadata_path) if metadata_path.exists() else None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path
