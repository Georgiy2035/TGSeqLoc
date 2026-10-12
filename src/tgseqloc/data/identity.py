"""Reproducible identity of a source file.

Kept apart from preparation so that a stage can fingerprint its inputs without
importing the graph stack, and therefore without torch.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def source_file_identity(path: str | Path) -> dict[str, Any]:
    """Return a reproducible path/stat/content identity for one source file."""

    source = Path(path)
    stat = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }
