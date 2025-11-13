"""Expose the CLI helpers when importing `studioeval` as a package."""

import sys as _sys
from contextlib import suppress
from importlib import import_module as _import_module

from .studioeval import exit, load_comm_config, main, setup_logging

__all__ = ("main", "setup_logging", "load_comm_config", "exit")

# When the repository root is treated as the `studioeval` package (common in
# local development), absolute imports like `import core` would normally fail
# because only the parent directory is on ``sys.path``.  We register aliases so
# that the traditional top-level modules remain importable without the package
# prefix, keeping existing tests and scripts working after the rename.
for _alias in (
    "config",
    "core",
    "eval_datasets",
    "implementations",
    "interfaces",
    "models",
    "utils",
    "tests",
):
    if _alias not in _sys.modules:
        with suppress(ModuleNotFoundError):
            _sys.modules[_alias] = _import_module(f".{_alias}", __name__)
