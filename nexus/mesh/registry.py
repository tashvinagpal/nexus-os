"""Devices paired into this NEXUS. Pairing itself (QR code / short code on the
LAN, confirmed on both screens) is a later milestone; it will end in add()."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from nexus.security.identity import b64d, device_id_for
from nexus.system.device import DeviceRole


@dataclass(frozen=True)
class DeviceRecord:
    device_id: str
    public_key: str  # base64url raw Ed25519
    name: str
    role: DeviceRole
    paired_at: str
    revoked: bool = False


class DeviceRegistry:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._devices: dict[str, DeviceRecord] = {}
        if path is not None and path.exists():
            for raw in json.loads(path.read_text(encoding="utf-8")):
                raw["role"] = DeviceRole(raw["role"])
                self._add(DeviceRecord(**raw))

    def add(self, record: DeviceRecord) -> None:
        self._add(record)
        self._save()

    def get(self, device_id: str) -> DeviceRecord | None:
        return self._devices.get(device_id)

    def revoke(self, device_id: str) -> bool:
        record = self._devices.get(device_id)
        if record is None:
            return False
        self._devices[device_id] = replace(record, revoked=True)
        self._save()
        return True

    def all(self) -> list[DeviceRecord]:
        return list(self._devices.values())

    def _add(self, record: DeviceRecord) -> None:
        # A device id must be derived from its key: no claiming someone else's id.
        if device_id_for(b64d(record.public_key)) != record.device_id:
            raise ValueError(f"device id {record.device_id} does not match its public key")
        self._devices[record.device_id] = record

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [{**asdict(r), "role": r.role.value} for r in self._devices.values()]
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
