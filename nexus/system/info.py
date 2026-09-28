"""Live hardware information: CPU, memory, GPU, storage."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any, Iterable

import psutil

from nexus.system.platform import Platform

SECTIONS = ("cpu", "memory", "gpu", "storage")
_GB = 1024 ** 3


def snapshot(platform: Platform, sections: Iterable[str] = SECTIONS) -> dict[str, Any]:
    readers = {
        "cpu": lambda: cpu(platform),
        "memory": memory,
        "gpu": gpus,
        "storage": storage,
    }
    return {s: readers[s]() for s in sections}


def cpu(platform: Platform) -> dict[str, Any]:
    return {
        "name": platform.cpu_name(),
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(logical=True),
        "usage_percent": psutil.cpu_percent(interval=0.3),
    }


def memory() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / _GB, 1),
        "used_gb": round((vm.total - vm.available) / _GB, 1),
        "usage_percent": vm.percent,
    }


def storage() -> list[dict[str, Any]]:
    drives = []
    for part in psutil.disk_partitions(all=False):
        if "cdrom" in part.opts or not part.fstype:
            continue
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except OSError:  # e.g. an empty card reader
            continue
        drives.append({
            "mount": part.mountpoint,
            "filesystem": part.fstype,
            "total_gb": round(usage.total / _GB, 1),
            "used_gb": round(usage.used / _GB, 1),
            "usage_percent": usage.percent,
        })
    return drives


def gpus() -> list[dict[str, Any]]:
    """NVIDIA GPUs via nvidia-smi (ships with the driver on Windows and Linux)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    fields = "name,memory.total,memory.used,utilization.gpu,temperature.gpu"
    try:
        proc = subprocess.run(
            [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    found = []
    for line in proc.stdout.strip().splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) != 5:
            continue
        total, used, load, temp = (_num(c) for c in cols[1:])
        found.append({
            "name": cols[0],
            "vram_total_mb": total,
            "vram_used_mb": used,
            "vram_total_gb": round(total / 1024, 1) if total is not None else None,
            "vram_used_gb": round(used / 1024, 1) if used is not None else None,
            "usage_percent": load,
            "temperature_c": temp,
        })
    return found


def _num(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:  # nvidia-smi prints "[N/A]" for unsupported fields
        return None


def format_report(data: dict[str, Any]) -> str:
    lines = []
    if "cpu" in data:
        c = data["cpu"]
        lines.append(f"CPU      {c['name']}, {c['cores']} cores / {c['threads']} threads, "
                     f"{c['usage_percent']:.0f}% busy")
    if "memory" in data:
        m = data["memory"]
        lines.append(f"Memory   {m['used_gb']} of {m['total_gb']} GB used ({m['usage_percent']:.0f}%)")
    if "gpu" in data:
        if not data["gpu"]:
            lines.append("GPU      no NVIDIA GPU detected (nvidia-smi not available)")
        for g in data["gpu"]:
            extra = []
            if g["vram_total_gb"] is not None and g["vram_used_gb"] is not None:
                extra.append(f"{g['vram_used_gb']} of {g['vram_total_gb']} GB VRAM")
            if g["usage_percent"] is not None:
                extra.append(f"{g['usage_percent']:.0f}% busy")
            if g["temperature_c"] is not None:
                extra.append(f"{g['temperature_c']:.0f} C")
            lines.append(f"GPU      {g['name']}" + (", " + ", ".join(extra) if extra else ""))
    if "storage" in data:
        for d in data["storage"]:
            lines.append(f"Storage  {d['mount']} {d['used_gb']:.0f} of {d['total_gb']:.0f} GB used "
                         f"({d['usage_percent']:.0f}%)")
    return "\n".join(lines)
