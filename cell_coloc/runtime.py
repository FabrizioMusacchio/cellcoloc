"""Runtime environment helpers for interactive desktop execution.

This module keeps user-facing scripts lightweight by centralizing small
environment workarounds that are only needed when default cache or config
locations are not writable. In normal local use nothing is redirected.

author: Fabrizio Musacchio
date: May/June 2026
"""
# %% IMPORTS
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import uuid

try:
    import appdirs
except ImportError:  # pragma: no cover - optional import guard
    appdirs = None

# %% RUNTIME ENVIRONMENT HELPERS
def get_runtime_cache_root() -> Path:
    """Return the temporary fallback directory used for runtime caches.

    The directory itself is only used when one of the default library cache or
    config locations turns out to be unavailable for writing.
    """

    return Path(tempfile.gettempdir()) / "cell_coloc_runtime_cache"


def _ensure_directory_and_probe_write(directory: Path) -> bool:
    """Return ``True`` when a directory can be created and written to."""

    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe_path = directory / f".cell_coloc_write_probe_{uuid.uuid4().hex}"
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink()
        return True
    except OSError:
        return False


def _get_default_matplotlib_dir() -> Path:
    """Return the default Matplotlib config directory without importing it."""

    if os.environ.get("MPLCONFIGDIR"):
        return Path(os.environ["MPLCONFIGDIR"]).expanduser()
    return Path.home() / ".matplotlib"


def _get_default_xdg_cache_dir() -> Path:
    """Return the default XDG-style cache directory."""

    if os.environ.get("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]).expanduser()
    return Path.home() / ".cache"


def _get_default_napari_settings_path() -> Path:
    """Return napari's default settings file path without importing napari."""

    if os.environ.get("NAPARI_CONFIG"):
        return Path(os.environ["NAPARI_CONFIG"]).expanduser()

    if appdirs is not None:
        config_dir = Path(appdirs.user_config_dir("napari", "napari")).expanduser()
    else:
        config_dir = Path.home() / ".config" / "napari"

    return config_dir / "settings.yaml"


def _set_env_if_fallback_needed(env_name: str, target_path: Path, needs_file_parent: bool = False) -> bool:
    """Set an environment variable to a temporary fallback only when needed.

    Parameters
    ----------
    env_name:
        Name of the environment variable to set, for example ``MPLCONFIGDIR``.
    target_path:
        Intended default directory or file path for the target library.
    needs_file_parent:
        Whether ``target_path`` points to a file path whose parent directory
        should be probed instead of the file itself.

    Returns
    -------
    bool
        ``True`` when a fallback path had to be installed, otherwise ``False``.
    """

    probe_directory = target_path.parent if needs_file_parent else target_path
    if _ensure_directory_and_probe_write(probe_directory):
        return False

    fallback_root = get_runtime_cache_root()
    fallback_root.mkdir(parents=True, exist_ok=True)
    fallback_target = fallback_root / env_name.lower()
    if needs_file_parent:
        fallback_target.parent.mkdir(parents=True, exist_ok=True)
    else:
        fallback_target.mkdir(parents=True, exist_ok=True)

    if env_name == "NAPARI_CONFIG":
        fallback_target = fallback_root / "napari" / "settings.yaml"
        fallback_target.parent.mkdir(parents=True, exist_ok=True)
    elif env_name == "MPLCONFIGDIR":
        fallback_target = fallback_root / "matplotlib"
        fallback_target.mkdir(parents=True, exist_ok=True)
    elif env_name == "NUMBA_CACHE_DIR":
        fallback_target = fallback_root / "numba"
        fallback_target.mkdir(parents=True, exist_ok=True)
    elif env_name == "XDG_CACHE_HOME":
        fallback_target = fallback_root / "xdg_cache"
        fallback_target.mkdir(parents=True, exist_ok=True)

    os.environ[env_name] = str(fallback_target)
    return True


def prepare_runtime_environment() -> dict[str, str]:
    """Prepare optional fallback cache locations for desktop library imports.

    The function checks whether the usual per-user config and cache locations
    for Matplotlib, napari, and related tooling are writable. Only if a path is
    unavailable will it redirect the corresponding environment variable to a
    temporary fallback under the system temp directory.

    Returns
    -------
    dict[str, str]
        Mapping of environment variables that were redirected to fallback paths.
        An empty mapping means the default user locations were usable.
    """

    applied_fallbacks: dict[str, str] = {}

    if _set_env_if_fallback_needed("MPLCONFIGDIR", _get_default_matplotlib_dir()):
        applied_fallbacks["MPLCONFIGDIR"] = os.environ["MPLCONFIGDIR"]

    xdg_cache_dir = _get_default_xdg_cache_dir()
    if _set_env_if_fallback_needed("XDG_CACHE_HOME", xdg_cache_dir):
        applied_fallbacks["XDG_CACHE_HOME"] = os.environ["XDG_CACHE_HOME"]

    numba_default_dir = Path(
        os.environ.get(
            "NUMBA_CACHE_DIR",
            str(Path(os.environ.get("XDG_CACHE_HOME", str(xdg_cache_dir))).expanduser() / "numba"),
        )
    ).expanduser()
    if _set_env_if_fallback_needed("NUMBA_CACHE_DIR", numba_default_dir):
        applied_fallbacks["NUMBA_CACHE_DIR"] = os.environ["NUMBA_CACHE_DIR"]

    if _set_env_if_fallback_needed("NAPARI_CONFIG", _get_default_napari_settings_path(), needs_file_parent=True):
        applied_fallbacks["NAPARI_CONFIG"] = os.environ["NAPARI_CONFIG"]

    return applied_fallbacks
# %% END