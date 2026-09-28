"""Read Windows shortcut files (.lnk, .url) as data. Nothing is executed or resolved via the shell.

.lnk format: [MS-SHLLINK]. We read the target path (LinkInfo, or the
environment-variable block), the arguments and the working directory.
"""

from __future__ import annotations

import os
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path

_CLSID = uuid.UUID("00021401-0000-0000-c000-000000000046").bytes_le
_HAS_IDLIST, _HAS_LINKINFO, _HAS_NAME, _HAS_RELPATH, _HAS_WORKDIR, _HAS_ARGS, _HAS_ICON, _UNICODE = (
    0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80)
_ENV_BLOCK, _DARWIN_BLOCK = 0xA0000001, 0xA0000006


@dataclass(frozen=True)
class Shortcut:
    target: str | None          # None for shell items (File Explorer, Control Panel) and some MSI shortcuts
    arguments: str = ""
    working_dir: str | None = None
    advertised: bool = False    # MSI "advertised" shortcut: only the shortcut itself knows how to start it


def _ansi(data: bytes, start: int) -> str:
    end = data.find(b"\x00", start)
    raw = data[start:end if end >= 0 else len(data)]
    try:
        return raw.decode("mbcs")
    except LookupError:  # not on Windows
        return raw.decode("latin-1")


def _unicode(data: bytes, start: int, max_chars: int | None = None) -> str:
    end = start
    limit = len(data) if max_chars is None else min(len(data), start + 2 * max_chars)
    while end + 1 < limit and data[end:end + 2] != b"\x00\x00":
        end += 2
    return data[start:end].decode("utf-16-le", "replace")


def parse_lnk(data: bytes, location: Path | None = None) -> Shortcut:
    if len(data) < 76 or struct.unpack_from("<I", data, 0)[0] != 0x4C or data[4:20] != _CLSID:
        raise ValueError("not a shell link")
    flags = struct.unpack_from("<I", data, 20)[0]
    pos = 76
    if flags & _HAS_IDLIST:
        pos += 2 + struct.unpack_from("<H", data, pos)[0]

    target = None
    if flags & _HAS_LINKINFO:
        size, header, li_flags, _vol, base_off, _net, suffix_off = struct.unpack_from("<7I", data, pos)
        if li_flags & 0x1:  # VolumeIDAndLocalBasePath
            base_u = suffix_u = 0
            if header >= 0x24:
                base_u, suffix_u = struct.unpack_from("<2I", data, pos + 28)
            base = _unicode(data, pos + base_u) if base_u else _ansi(data, pos + base_off)
            suffix = _unicode(data, pos + suffix_u) if suffix_u else _ansi(data, pos + suffix_off)
            target = base + suffix
        pos += size

    strings: dict[int, str] = {}
    for bit in (_HAS_NAME, _HAS_RELPATH, _HAS_WORKDIR, _HAS_ARGS, _HAS_ICON):
        if flags & bit:
            count = struct.unpack_from("<H", data, pos)[0]
            pos += 2
            if flags & _UNICODE:
                strings[bit] = data[pos:pos + 2 * count].decode("utf-16-le", "replace")
                pos += 2 * count
            else:
                strings[bit] = data[pos:pos + count].decode("latin-1")
                pos += count

    advertised = False
    while pos + 8 <= len(data):
        block_size, signature = struct.unpack_from("<2I", data, pos)
        if block_size < 8:
            break
        if signature == _ENV_BLOCK and block_size >= 0x314 and not target:
            target = os.path.expandvars(_unicode(data, pos + 8 + 260, 260)) or None
        elif signature == _DARWIN_BLOCK:
            advertised = True
        pos += block_size

    if not target and _HAS_RELPATH in strings and location is not None:
        candidate = (location.parent / strings[_HAS_RELPATH]).resolve()
        target = str(candidate) if candidate.exists() else None
    return Shortcut(target=target or None, arguments=strings.get(_HAS_ARGS, "").strip(),
                    working_dir=strings.get(_HAS_WORKDIR) or None, advertised=advertised)


def read_lnk(path: Path) -> Shortcut | None:
    try:
        return parse_lnk(path.read_bytes(), path)
    except (OSError, ValueError, struct.error):
        return None


def read_url(path: Path) -> str | None:
    """The URL= line of an Internet shortcut."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    for line in text.splitlines():
        if line.strip().lower().startswith("url="):
            return line.split("=", 1)[1].strip() or None
    return None
