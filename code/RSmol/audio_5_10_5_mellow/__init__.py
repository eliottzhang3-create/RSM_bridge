"""Isolated audio-audit helpers for the fixed 5-10-5 Mellow route.

This package deliberately contains no model implementation and never imports
the legacy text, MeSH, or Parcae variants.  Remote Mellow/HTSAT source trees
are supplied to the Stage 2 CLI at runtime.
"""

from .manifest import (
    AUDIO_EXTENSIONS,
    ManifestAuditError,
    build_audio_index,
    build_reasonaqa_manifests,
    canonical_json_hash,
    load_reasonaqa_json,
    resolve_audio_path,
)

__all__ = [
    "AUDIO_EXTENSIONS",
    "ManifestAuditError",
    "build_audio_index",
    "build_reasonaqa_manifests",
    "canonical_json_hash",
    "load_reasonaqa_json",
    "resolve_audio_path",
]
