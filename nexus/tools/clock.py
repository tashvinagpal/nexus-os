from __future__ import annotations

from datetime import datetime
from typing import Callable

from nexus.tools.base import Risk, Tool, ToolResult


def make_clock_tool(now: Callable[[], datetime] = lambda: datetime.now().astimezone()) -> Tool:
    def run() -> ToolResult:
        t = now()
        text = f"It's {t:%H:%M} on {t:%A}, {t.day} {t:%B %Y}."
        return ToolResult(True, text, {"iso": t.isoformat(timespec="seconds")}, verified=True)

    return Tool(name="get_datetime", description="Tell the current local time and date.",
                risk=Risk.SAFE, params=(), run=run)
