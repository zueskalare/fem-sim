"""Consolidated FreeFEM binary detection.

Used by both the framework backend (backends/freefem.py) and
the pixel-to-FEM bridge (pixel_to_fem.py).
"""

from __future__ import annotations

import shutil
from pathlib import Path

# macOS app bundle search path
_MACOS_APP_GLOB = "/Applications/FreeFem++.app/Contents/ff-*/bin/FreeFem++"


def find_freefem_binary(override: str | None = None) -> str:
    """Locate the FreeFEM++ binary.

    Search order:
      1. Explicit override (if provided)
      2. PATH lookup for FreeFem++, freefem++, ff-mpirun
      3. macOS app bundle at /Applications/FreeFem++.app/...

    Parameters
    ----------
    override : str, optional
        Explicit binary name or path. If given and found, returned immediately.

    Returns
    -------
    str
        Path or name of the FreeFEM++ binary.

    Raises
    ------
    FileNotFoundError
        If no FreeFEM++ binary can be found.
    """
    if override:
        found = shutil.which(override)
        if found:
            return found
        if Path(override).exists():
            return override

    # Check PATH for common names.
    for name in ("FreeFem++", "freefem++", "ff-mpirun"):
        found = shutil.which(name)
        if found:
            return found

    # macOS app bundle fallback.
    import glob as _glob
    candidates = sorted(_glob.glob(_MACOS_APP_GLOB))
    if candidates:
        return candidates[-1]  # newest version last when sorted

    raise FileNotFoundError(
        "FreeFem++ not found. Install FreeFEM or provide the binary path."
    )
