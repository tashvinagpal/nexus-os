"""Platform layer: the one place that knows which OS NEXUS runs on.

Tools and services call a Platform, never Windows or Linux APIs directly.
Moving NEXUS to its own Linux base means adding a LinuxPlatform here.
"""

from __future__ import annotations

import sys

from .base import Platform


def current() -> Platform:
    if sys.platform == "win32":
        from .windows import WindowsPlatform

        return WindowsPlatform()
    raise NotImplementedError(
        f"NEXUS has no platform backend for {sys.platform!r} yet "
        "(Linux support is planned for v0.9)")


__all__ = ["Platform", "current"]
