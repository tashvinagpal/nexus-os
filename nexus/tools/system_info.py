from __future__ import annotations

from nexus.system import info
from nexus.system.platform import Platform
from nexus.tools.base import Param, Risk, Tool, ToolResult


def make_system_info_tool(platform: Platform) -> Tool:
    def run(section: str) -> ToolResult:
        sections = info.SECTIONS if section == "all" else (section,)
        data = info.snapshot(platform, sections)
        return ToolResult(ok=True, message=info.format_report(data), data=data, verified=True)

    return Tool(
        name="get_system_info",
        description="Report live CPU, memory, GPU (VRAM, load, temperature) and storage usage.",
        risk=Risk.SAFE,
        params=(Param("section", "str", "which part to report", required=False,
                      default="all", choices=("all", *info.SECTIONS)),),
        run=run,
    )
