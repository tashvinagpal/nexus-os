"""The structured tool layer. Every action NEXUS can take is a Tool defined here."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nexus.ai.memory import MemoryStore
from nexus.system.platform import Platform
from nexus.tools.base import (Param, Risk, Tool, ToolRegistry, ToolResult,
                              ValidationError)
from nexus.tools.clock import make_clock_tool
from nexus.tools.files import make_file_tools
from nexus.tools.memory import make_memory_tools
from nexus.tools.system_info import make_system_info_tool
from nexus.tools.web import make_web_tools

if TYPE_CHECKING:
    from nexus.system.apps.service import AppService


def default_tools(platform: Platform, memory: MemoryStore | None = None,
                  apps: AppService | None = None) -> list[Tool]:
    tools = [
        make_system_info_tool(platform),
        make_clock_tool(),
        *make_web_tools(platform),
        *make_file_tools(),
    ]
    if memory is not None:
        tools += make_memory_tools(memory)
    if apps is not None:
        from nexus.tools.apps import make_app_tools

        tools += make_app_tools(apps)
    return tools


__all__ = ["Param", "Risk", "Tool", "ToolRegistry", "ToolResult", "ValidationError",
           "default_tools"]
