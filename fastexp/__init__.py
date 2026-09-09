"""Fastexp package root (artifact-first research platform).

CLI:    `python -m fastexp.cli simulate ...`
Server: `python -m fastexp.server`
SDK:    `import fastexp as fe; fe.simulate(...)`

The simulation engine modules live at the repository root and are imported as
top-level modules; this package hosts the user-facing CLI, Web Studio server,
and the notebook SDK. SDK access is lazy so that `import fastexp` stays cheap
(the CLI and server must not pay for pandas/IPython import cost).
"""
__version__ = "1.0.0"

__all__ = ["simulate", "batch", "Run"]


def __getattr__(name):
    if name in ("simulate", "batch", "Run"):
        from . import sdk
        return getattr(sdk, name)
    raise AttributeError(f"module 'fastexp' has no attribute {name!r}")
