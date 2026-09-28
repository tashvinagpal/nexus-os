from __future__ import annotations

import ctypes
import json
import os
import platform as _stdlib_platform
import re
import subprocess
import uuid
from ctypes import wintypes
from pathlib import Path
from typing import Iterable

from nexus.system.platform.base import AudioInput, LaunchError, LaunchMethod, LaunchSpec

# Windows known-folder ids (correct even when Desktop/Documents are redirected to OneDrive).
_KNOWN_FOLDERS = {
    "home": "5E6C858F-0E22-4760-9AFE-EA3317B67173",
    "desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
    "documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
    "downloads": "374DE290-123F-4565-9164-39C4925E467B",
    "pictures": "33E28130-4E1E-4676-835A-98395C3BC3BB",
    "music": "4BD8D571-6D19-48D3-BE97-422220080E43",
    "videos": "18989B1D-99B5-455B-841C-AB7C74E4DDFC",
    "public_desktop": "C4AA340D-F20F-4863-AFEF-F87EF2E6BA25",
    "start_menu": "A77F5D77-2E2B-44C3-A6A2-ABA601054A51",
    "common_start_menu": "0139D44E-6AFE-49F2-8690-3DAFCAE6FFB8",
    "program_data": "62AB5D82-FDC1-4DC3-A9DD-070D1D495D97",
    "local_app_data": "F1B32785-6FBA-4FCF-9D55-7B8E7F157091",
}
_URI_SCHEMES = ("steam://", "com.epicgames.launcher://")
_AUMID = re.compile(r"^[A-Za-z0-9.\-]+_[a-z0-9]{13}![A-Za-z0-9.\-]+$")
_ERROR_ELEVATION_REQUIRED = 740


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8)]


def _guid(text: str) -> _GUID:
    u = uuid.UUID(text)
    g = _GUID(u.fields[0], u.fields[1], u.fields[2])
    g.Data4[:] = u.bytes[8:]
    return g


_WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def _user32():
    u = ctypes.WinDLL("user32", use_last_error=True)
    u.EnumWindows.argtypes = [_WNDENUMPROC, wintypes.LPARAM]
    u.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    u.IsWindowVisible.argtypes = [wintypes.HWND]
    u.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    u.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
    u.GetWindow.restype = wintypes.HWND
    u.IsIconic.argtypes = [wintypes.HWND]
    u.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    u.SetForegroundWindow.argtypes = [wintypes.HWND]
    u.GetForegroundWindow.restype = wintypes.HWND
    return u


# --- audio inputs: Core Audio (MMDevice) API, read-only - no device is ever opened here ---

_CLSID_MM_DEVICE_ENUMERATOR = "BCDE0395-E52F-467C-8E3D-C4579291692E"
_IID_IMM_DEVICE_ENUMERATOR = "A95664D2-9614-4F35-A746-DE8DB63617E6"
_E_CAPTURE = 1
_ROLE_CONSOLE, _ROLE_COMMUNICATIONS = 0, 2
_DEVICE_STATES = {0x1: "active", 0x2: "disabled", 0x8: "unplugged"}  # NOT_PRESENT (removed) is skipped
_RPC_E_CHANGED_MODE = -2147417850  # COM already initialised differently on this thread: just use it
_FORM_FACTORS = {4: "microphone", 5: "headset", 6: "handset", 2: "line_in", 10: "unknown"}
_PK_FRIENDLY_NAME = ("A45C254E-DF1C-4EFD-8020-67D146A850E0", 14)
_PK_ENUMERATOR = ("A45C254E-DF1C-4EFD-8020-67D146A850E0", 24)   # USB, HDAUDIO, BTHENUM, ROOT...
_PK_FORM_FACTOR = ("1DA5D803-D492-4EDD-8C23-E0C0FFEE7F0E", 0)
_PK_DEVICE_PATH = ("B3F8FA53-0004-438E-9003-51A46E139BFC", 2)   # "{1}.BTHENUM\{0000110A-...}..."
_A2DP_SOURCE = "{0000110A-"  # Bluetooth "audio source" service: a phone streaming its sound to this PC


class _PROPERTYKEY(ctypes.Structure):
    _fields_ = [("fmtid", _GUID), ("pid", wintypes.DWORD)]


class _PROPVARIANT(ctypes.Structure):
    _fields_ = [("vt", ctypes.c_ushort), ("reserved", ctypes.c_ushort * 3), ("value", ctypes.c_void_p),
                ("extra", ctypes.c_void_p)]


def _com(obj: ctypes.c_void_p, index: int, *argtypes):
    """Method `index` of a COM object's vtable; a failed HRESULT raises OSError."""
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    fn = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, *argtypes)(vtable[index])
    return lambda *args: fn(obj, *args)


def _com_release(obj: ctypes.c_void_p) -> None:
    if obj:
        vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable[2])(obj)


def _ole32():
    ole = ctypes.WinDLL("ole32")
    ole.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    ole.CoInitializeEx.restype = ctypes.c_long
    ole.CoCreateInstance.argtypes = [ctypes.POINTER(_GUID), ctypes.c_void_p, wintypes.DWORD,
                                     ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
    ole.CoCreateInstance.restype = ctypes.HRESULT
    ole.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    ole.PropVariantClear.argtypes = [ctypes.POINTER(_PROPVARIANT)]
    return ole


def _audio_bus(enumerator: str) -> str:
    enumerator = enumerator.upper()
    if enumerator in ("HDAUDIO", "INTELAUDIO", "ACPI", "PCI"):
        return "builtin"
    if enumerator == "USB":
        return "usb"
    if enumerator.startswith("BTH"):
        return "bluetooth"
    if enumerator == "ROOT":  # root-enumerated = a software driver (virtual cameras, virtual cables...)
        return "virtual"
    return "other"


_MIC_CONSENT = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone"


class WindowsPlatform:
    name = "windows"

    def microphone_allowed(self) -> bool | None:
        """Windows privacy switches: device-wide, this user, and "let desktop apps access your microphone"."""
        values = [self.read_registry(f"HKLM\\{_MIC_CONSENT}", "Value"),
                  self.read_registry(f"HKCU\\{_MIC_CONSENT}", "Value"),
                  self.read_registry(f"HKCU\\{_MIC_CONSENT}\\NonPackaged", "Value")]
        if any(v == "Deny" for v in values):
            return False
        return True if any(v == "Allow" for v in values) else None

    def audio_inputs(self) -> list[AudioInput]:
        ole = _ole32()
        hr = ole.CoInitializeEx(None, 0x0)  # COINIT_MULTITHREADED
        if hr < 0 and hr != _RPC_E_CHANGED_MODE:
            raise OSError(f"COM initialisation failed (0x{hr & 0xFFFFFFFF:08X})")
        enumerator = ctypes.c_void_p()
        try:
            ole.CoCreateInstance(ctypes.byref(_guid(_CLSID_MM_DEVICE_ENUMERATOR)), None, 0x17,  # CLSCTX_ALL
                                 ctypes.byref(_guid(_IID_IMM_DEVICE_ENUMERATOR)), ctypes.byref(enumerator))
            default = self._default_input(enumerator, _ROLE_CONSOLE, ole)
            communications = self._default_input(enumerator, _ROLE_COMMUNICATIONS, ole)
            collection = ctypes.c_void_p()
            _com(enumerator, 3, ctypes.c_int, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p))(
                _E_CAPTURE, 0x1 | 0x2 | 0x8, ctypes.byref(collection))  # active | disabled | unplugged
            inputs: list[AudioInput] = []
            try:
                count = wintypes.UINT()
                _com(collection, 3, ctypes.POINTER(wintypes.UINT))(ctypes.byref(count))
                for i in range(count.value):
                    device = ctypes.c_void_p()
                    try:
                        _com(collection, 4, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))(i, ctypes.byref(device))
                        inputs.append(self._audio_input(device, default, communications, ole))
                    except OSError:
                        continue  # unplugged while we were looking
                    finally:
                        _com_release(device)
            finally:
                _com_release(collection)
            return inputs
        finally:
            _com_release(enumerator)
            if hr >= 0:
                ole.CoUninitialize()

    @staticmethod
    def _device_id(device: ctypes.c_void_p, ole) -> str:
        raw = ctypes.c_void_p()
        _com(device, 5, ctypes.POINTER(ctypes.c_void_p))(ctypes.byref(raw))  # IMMDevice::GetId
        try:
            return ctypes.wstring_at(raw.value) if raw.value else ""
        finally:
            ole.CoTaskMemFree(raw)

    def _default_input(self, enumerator: ctypes.c_void_p, role: int, ole) -> str | None:
        device = ctypes.c_void_p()
        try:
            _com(enumerator, 4, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))(
                _E_CAPTURE, role, ctypes.byref(device))
            return self._device_id(device, ole)
        except OSError:  # E_NOTFOUND: no active input at all
            return None
        finally:
            _com_release(device)

    def _audio_input(self, device: ctypes.c_void_p, default: str | None, communications: str | None,
                     ole) -> AudioInput:
        endpoint_id = self._device_id(device, ole)
        state = wintypes.DWORD()
        _com(device, 6, ctypes.POINTER(wintypes.DWORD))(ctypes.byref(state))  # IMMDevice::GetState
        store = ctypes.c_void_p()
        _com(device, 4, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p))(0, ctypes.byref(store))  # STGM_READ
        try:
            def prop(key: tuple[str, int]) -> str | int | None:
                pk = _PROPERTYKEY(_guid(key[0]), key[1])
                pv = _PROPVARIANT()
                try:
                    _com(store, 5, ctypes.POINTER(_PROPERTYKEY), ctypes.POINTER(_PROPVARIANT))(
                        ctypes.byref(pk), ctypes.byref(pv))  # IPropertyStore::GetValue
                except OSError:
                    return None
                try:
                    if pv.vt == 31:  # VT_LPWSTR
                        return ctypes.wstring_at(pv.value) if pv.value else ""
                    if pv.vt == 19:  # VT_UI4
                        return (pv.value or 0) & 0xFFFFFFFF
                    return None
                finally:
                    ole.PropVariantClear(ctypes.byref(pv))

            name = str(prop(_PK_FRIENDLY_NAME) or "Unknown input")
            enumerator = str(prop(_PK_ENUMERATOR) or "")
            path = str(prop(_PK_DEVICE_PATH) or "").upper()
            form = prop(_PK_FORM_FACTOR)
        finally:
            _com_release(store)
        form_factor = _FORM_FACTORS.get(form if isinstance(form, int) else -1, "other")
        bus = _audio_bus(enumerator)
        # A codec input with no microphone/line form factor is a mix of what the PC plays ("Stereo Mix").
        loopback = bus == "builtin" and form_factor == "unknown"
        # This PC as the speaker of a phone: the phone's call (hands-free "HF Audio") or its music (A2DP).
        remote = bus == "bluetooth" and (_A2DP_SOURCE in path or (
            enumerator.upper() == "BTHHFENUM" and form_factor not in ("headset", "handset")))
        return AudioInput(id=endpoint_id, name=name, state=_DEVICE_STATES.get(state.value & 0xF, "unknown"),
                          form_factor=form_factor, bus=bus, loopback=loopback, remote=remote,
                          default=endpoint_id == default, default_communications=endpoint_id == communications)

    def protected_paths(self) -> list[Path]:
        env = os.environ
        drive = env.get("SystemDrive", "C:") + "\\"
        candidates = [
            env.get("SystemRoot", drive + "Windows"),
            env.get("ProgramFiles"),
            env.get("ProgramFiles(x86)"),
            env.get("ProgramW6432"),
            env.get("ProgramData"),
            str(Path.home() / "AppData"),  # installed apps' settings and data
            drive + "Recovery",
            drive + "System Volume Information",
            drive + "$Recycle.Bin",
            drive + "$WinREAgent",
            drive + "Boot",
            drive + "Users\\Default",
        ]
        return [Path(p) for p in dict.fromkeys(c for c in candidates if c)]

    def safe_roots(self) -> list[Path]:
        return [Path.home()]

    def open_uri(self, uri: str) -> None:
        os.startfile(uri)  # type: ignore[attr-defined]  # Windows-only API

    def is_elevated(self) -> bool:
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            return False

    def cpu_name(self) -> str:
        name = self.read_registry(r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0", "ProcessorNameString")
        return (name or _stdlib_platform.processor() or "unknown CPU").strip()

    # --- apps and games ----------------------------------------------------------------

    def known_folders(self) -> dict[str, Path]:
        shell32 = ctypes.WinDLL("shell32")
        ole32 = ctypes.WinDLL("ole32")
        shell32.SHGetKnownFolderPath.argtypes = [ctypes.POINTER(_GUID), wintypes.DWORD, wintypes.HANDLE,
                                                 ctypes.POINTER(ctypes.c_wchar_p)]
        ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
        folders: dict[str, Path] = {}
        for role, fid in _KNOWN_FOLDERS.items():
            ptr = ctypes.c_wchar_p()
            if shell32.SHGetKnownFolderPath(ctypes.byref(_guid(fid)), 0, None, ctypes.byref(ptr)) == 0:
                folders[role] = Path(ptr.value or "")
            ole32.CoTaskMemFree(ctypes.cast(ptr, ctypes.c_void_p))
        folders["temp"] = Path(os.environ.get("TEMP", str(folders.get("local_app_data", Path.home()) / "Temp")))
        return folders

    def read_registry(self, key: str, value: str = "") -> str | None:
        import winreg

        hive, _, sub = key.partition("\\")
        root = {"HKLM": winreg.HKEY_LOCAL_MACHINE, "HKCU": winreg.HKEY_CURRENT_USER}.get(hive)
        if root is None:
            return None
        try:
            with winreg.OpenKey(root, sub) as k:
                data, kind = winreg.QueryValueEx(k, value)
        except OSError:
            return None
        if kind == winreg.REG_EXPAND_SZ:
            data = os.path.expandvars(data)
        return str(data) if isinstance(data, (str, int)) else None

    def registry_subkeys(self, key: str) -> list[str]:
        import winreg

        hive, _, sub = key.partition("\\")
        root = {"HKLM": winreg.HKEY_LOCAL_MACHINE, "HKCU": winreg.HKEY_CURRENT_USER}.get(hive)
        names: list[str] = []
        if root is None:
            return names
        try:
            with winreg.OpenKey(root, sub) as k:
                i = 0
                while True:
                    try:
                        names.append(winreg.EnumKey(k, i))
                    except OSError:
                        break
                    i += 1
        except OSError:
            pass
        return names

    def list_store_apps(self) -> list[tuple[str, str]]:
        # A fixed, read-only query - never built from user or model text.
        ps = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
        command = ("[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
                   "Get-StartApps | Select-Object Name, AppID | ConvertTo-Json -Compress")
        try:
            proc = subprocess.run([str(ps), "-NoProfile", "-NonInteractive", "-Command", command],
                                  capture_output=True, timeout=30, check=False,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            items = json.loads(proc.stdout.decode("utf-8", "replace") or "[]")
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            return []
        if isinstance(items, dict):
            items = [items]
        return [(i["Name"], i["AppID"]) for i in items
                if isinstance(i, dict) and _AUMID.match(str(i.get("AppID", "")))]

    def launch(self, spec: LaunchSpec) -> None:
        try:
            if spec.method is LaunchMethod.EXECUTABLE:
                exe = Path(spec.target)
                if not exe.is_absolute() or exe.suffix.lower() != ".exe" or not exe.is_file():
                    raise LaunchError(f"{exe} is not an installed program")
                command = subprocess.list2cmdline([str(exe)]) + (f" {spec.arguments}" if spec.arguments else "")
                subprocess.Popen(command, cwd=spec.working_dir or str(exe.parent), close_fds=True,
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
            elif spec.method is LaunchMethod.URI:
                if not spec.target.lower().startswith(_URI_SCHEMES):
                    raise LaunchError(f"unsupported launcher address {spec.target!r}")
                os.startfile(spec.target)  # type: ignore[attr-defined]
            elif spec.method is LaunchMethod.PACKAGED:
                if not _AUMID.match(spec.target):
                    raise LaunchError(f"invalid packaged app id {spec.target!r}")
                explorer = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "explorer.exe"
                subprocess.Popen([str(explorer), f"shell:AppsFolder\\{spec.target}"], close_fds=True)
            elif spec.method is LaunchMethod.SHORTCUT:
                lnk = Path(spec.target)
                if lnk.suffix.lower() != ".lnk" or not lnk.is_file():
                    raise LaunchError(f"{lnk} is not a shortcut")
                os.startfile(str(lnk))  # type: ignore[attr-defined]
        except LaunchError:
            raise
        except OSError as exc:
            if getattr(exc, "winerror", None) == _ERROR_ELEVATION_REQUIRED:
                raise LaunchError("it needs administrator rights, and NEXUS never starts programs as "
                                  "administrator") from None
            raise LaunchError(exc.strerror or str(exc)) from None

    def open_folder(self, path: str) -> None:
        if path.startswith("shell:"):
            explorer = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "explorer.exe"
            subprocess.Popen([str(explorer), path], close_fds=True)
            return
        # The "explore" verb only exists for folders, so a file (e.g. an .exe) can never be run here.
        os.startfile(path, "explore")  # type: ignore[attr-defined]

    def _windows_of(self, pids: set[int]) -> list[int]:
        user32 = _user32()
        found: list[int] = []

        def callback(hwnd, _):
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if (pid.value in pids and user32.IsWindowVisible(hwnd) and user32.GetWindowTextLengthW(hwnd) > 0
                    and not user32.GetWindow(hwnd, 4)):  # 4 = GW_OWNER: skip dialogs owned by other windows
                found.append(hwnd)
            return True

        user32.EnumWindows(_WNDENUMPROC(callback), 0)
        return found

    def window_pids(self, pids: Iterable[int]) -> set[int]:
        wanted = set(pids)
        if not wanted:
            return set()
        user32 = _user32()
        owners = set()
        for hwnd in self._windows_of(wanted):
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            owners.add(pid.value)
        return owners

    def focus_window(self, pids: Iterable[int]) -> bool:
        windows = self._windows_of(set(pids))
        if not windows:
            return False
        user32 = _user32()
        hwnd = windows[0]
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
        return user32.GetForegroundWindow() == hwnd

    def close_windows(self, pids: Iterable[int]) -> int:
        windows = self._windows_of(set(pids))
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        asked = 0
        for hwnd in windows:
            if user32.PostMessageW(hwnd, 0x0010, 0, 0):  # WM_CLOSE: the app decides (it may ask to save)
                asked += 1
        return asked

    def requires_elevation(self, exe: Path) -> bool | None:
        """Reads the program's embedded manifest (as data - nothing is executed)."""
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.LoadLibraryExW.argtypes = [wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD]
        k32.LoadLibraryExW.restype = wintypes.HMODULE
        k32.FindResourceW.argtypes = [wintypes.HMODULE, ctypes.c_void_p, ctypes.c_void_p]
        k32.FindResourceW.restype = ctypes.c_void_p
        k32.SizeofResource.argtypes = [wintypes.HMODULE, ctypes.c_void_p]
        k32.LoadResource.argtypes = [wintypes.HMODULE, ctypes.c_void_p]
        k32.LoadResource.restype = ctypes.c_void_p
        k32.LockResource.argtypes = [ctypes.c_void_p]
        k32.LockResource.restype = ctypes.c_void_p
        k32.FreeLibrary.argtypes = [wintypes.HMODULE]
        module = k32.LoadLibraryExW(str(exe), None, 0x2 | 0x20)  # AS_DATAFILE | AS_IMAGE_RESOURCE
        if not module:
            return None
        try:
            res = k32.FindResourceW(module, ctypes.c_void_p(1), ctypes.c_void_p(24))  # RT_MANIFEST, id 1
            if not res:
                return False
            size = k32.SizeofResource(module, res)
            ptr = k32.LockResource(k32.LoadResource(module, res))
            return bool(ptr) and b"requireAdministrator" in ctypes.string_at(ptr, size)
        finally:
            k32.FreeLibrary(module)
