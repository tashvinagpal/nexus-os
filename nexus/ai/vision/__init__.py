"""Screen understanding interface (v0.5). Vision models load only in the VISION
state and never while game mode is on."""

from __future__ import annotations

from typing import Protocol

from nexus.ai.providers.base import ProviderInfo


class VisionProvider(Protocol):
    info: ProviderInfo

    def describe(self, image_png: bytes, *, question: str) -> str: ...
