"""Pytest configuration for CellColoc."""

from __future__ import annotations

import os

# Avoid numba/napari cache issues during test collection in editable envs.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
