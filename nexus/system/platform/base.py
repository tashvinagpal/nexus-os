from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


class LaunchMethod(str, enum.Enum):
    EXECUTABLE = "executable"  # a discovered .exe with the arguments from its shortcut
    SHORTCUT = "shortcut"      # a discovered shortcut with no plain target (shell items like File Explorer)
    URI = "uri"                # a launcher URI: steam://rungameid/..., com.epicgames.launcher://...
    PACKAGED = "packaged"      # a Store/packaged app, by its AppUserModelID


@dataclass(frozen=True)
class LaunchSpec:
    """How to start something that discovery found. Never built from model text."""

    method: LaunchMethod
    target: str                   # exe path | .lnk path | URI | AppUserModelID
    arguments: str = ""           # only from the discovered shortcut
    working_dir: str | None = None


class LaunchError(OSError):
    """The OS refused to start it (missing file, needs administrator rights, ...)."""


@dataclass(frozen=True)
class AudioInput:
    """An audio capture endpoint as the OS reports it - facts only, read-only.

    NEXUS decides what is a usable microphone in nexus/system/audio; the
    platform only translates its own terms into these fields.
    """

    id: str                        # stable OS endpoint id
    name: str                      # "Microphone (Realtek(R) Audio)"
    state: str = "active"          # active | disabled | unplugged
    form_factor: str = "unknown"   # microphone | headset | handset | line_in | unknown | other
    bus: str = "other"             # builtin | usb | bluetooth | virtual | other
    loopback: bool = False         # records what the PC plays ("Stereo Mix"), not a microphone
    remote: bool = False           # audio from another device (a phone's call/music over Bluetooth)
    default: bool = False          # the OS default input
    default_communications: bool = False  # the OS default input for calls/voice


class Platform(Protocol):
    name: str

    def protected_paths(self) -> list[Path]:
        """Folders NEXUS must never modify (OS files, installed programs, app data)."""
        ...

    def safe_roots(self) -> list[Path]:
        """Default folders where NEXUS may write without asking."""
        ...

    def open_uri(self, uri: str) -> None:
        """Hand a web address to the default browser. Raises OSError on failure."""
        ...

    def is_elevated(self) -> bool:
        """True if running with administrator/root rights."""
        ...

    def cpu_name(self) -> str:
        ...

    # --- apps and games (v0.1 subsystem 4) ------------------------------------------

    def known_folders(self) -> dict[str, Path]:
        """User and system folders by role: home, desktop, documents, downloads, pictures, music,
        videos, public_desktop, start_menu, common_start_menu, program_data, local_app_data, temp."""
        ...

    def read_registry(self, key: str, value: str = "") -> str | None:
        """A registry string (Windows); None elsewhere or if missing. value "" = default value."""
        ...

    def registry_subkeys(self, key: str) -> list[str]:
        ...

    def list_store_apps(self) -> list[tuple[str, str]]:
        """Packaged (Store) apps as (display name, AppUserModelID)."""
        ...

    def launch(self, spec: LaunchSpec) -> None:
        """Start a discovered app. Raises LaunchError. Never elevates."""
        ...

    def open_folder(self, path: str) -> None:
        """Show a folder in the file manager. Fails for anything that isn't a folder."""
        ...

    def window_pids(self, pids: Iterable[int]) -> set[int]:
        """Which of these processes own a visible top-level window."""
        ...

    def focus_window(self, pids: Iterable[int]) -> bool:
        """Bring a window of these processes to the front. True if it is now in front."""
        ...

    def close_windows(self, pids: Iterable[int]) -> int:
        """Politely ask these processes' visible windows to close (like clicking X). Returns how many were asked."""
        ...

    def requires_elevation(self, exe: Path) -> bool | None:
        """True if the program demands administrator rights; None if unknown."""
        ...

    def audio_inputs(self) -> list[AudioInput]:
        """Every audio capture endpoint that is active, disabled or unplugged (not removed ones).

        Called again whenever NEXUS needs a fresh view, so devices plugged in
        after start-up appear without a restart. Never opens a device.
        """
        ...

    def microphone_allowed(self) -> bool | None:
        """False if the OS privacy settings block microphone access for this app; None if unknown."""
        ...
