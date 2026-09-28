from __future__ import annotations

from pathlib import Path

from nexus.tools.base import Param, Risk, Tool, ToolResult


def create_folder(path: Path) -> ToolResult:
    data = {"path": str(path)}
    if path.is_dir():
        return ToolResult(True, f"The folder {path} already exists.", data, verified=True)
    if path.exists():
        return ToolResult(False, f"{path} already exists and is a file, not a folder.", data)
    try:
        path.mkdir(parents=True)
    except OSError as exc:
        return ToolResult(False, f"Could not create {path}: {exc.strerror or exc}", data)
    if not path.is_dir():
        return ToolResult(False, f"The system reported success but {path} does not exist.", data, verified=False)
    return ToolResult(True, f"Created the folder {path}.", data, verified=True)


def make_file_tools() -> list[Tool]:
    return [
        Tool(
            name="create_folder",
            description="Create a folder (and any missing parent folders).",
            risk=Risk.SAFE,  # the policy escalates to CONFIRM outside your safe folders
            params=(Param("path", "path", "full path of the folder to create", writes=True),),
            run=create_folder,
            describe=lambda a: f"Create the folder {a['path']}",
        ),
    ]
