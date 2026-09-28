"""Identities: one NEXUS per user, one key pair per device.

A device id is derived from the device's public key, so no device can claim
another device's id without its private key. Every message between devices
is signed with that key (see nexus/mesh).

v0.1 stores the private key as a file in NEXUS's protected data folder
(tools can't read or write there). Hardware-backed storage (Windows DPAPI /
TPM, Android Keystore, iOS Secure Enclave) comes with the network layer.
"""

from __future__ import annotations

import base64
import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from nexus.system.device import DeviceRole


def b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def device_id_for(public_key: bytes) -> str:
    digest = hashlib.sha256(public_key).digest()
    return "nxd-" + base64.b32encode(digest).decode("ascii").lower().rstrip("=")[:26]


def verify_signature(public_key: bytes, data: bytes, signature: bytes) -> bool:
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, data)
        return True
    except (InvalidSignature, ValueError):
        return False


@dataclass(frozen=True)
class DeviceIdentity:
    """The public, shareable part of a device's identity."""

    device_id: str
    public_key: str  # base64url raw Ed25519 public key
    name: str
    role: DeviceRole


class DeviceKey:
    """This device's private identity. Never leaves the device."""

    def __init__(self, private_key: Ed25519PrivateKey, *, name: str, role: DeviceRole) -> None:
        self._key = private_key
        raw = private_key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.identity = DeviceIdentity(device_id_for(raw), b64e(raw), name, role)

    @property
    def device_id(self) -> str:
        return self.identity.device_id

    def sign(self, data: bytes) -> bytes:
        return self._key.sign(data)

    @classmethod
    def generate(cls, *, name: str, role: DeviceRole) -> DeviceKey:
        return cls(Ed25519PrivateKey.generate(), name=name, role=role)

    @classmethod
    def load_or_create(cls, path: Path, *, name: str, role: DeviceRole) -> DeviceKey:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            key = cls(Ed25519PrivateKey.from_private_bytes(b64d(data["private_key"])),
                      name=data["name"], role=DeviceRole(data["role"]))
            if key.device_id != data["device_id"]:
                raise RuntimeError(f"device identity file {path} is inconsistent; refusing to use it")
            return key
        key = cls.generate(name=name, role=role)
        raw = key._key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                     serialization.NoEncryption())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"device_id": key.device_id, "name": name, "role": role.value,
                                    "private_key": b64e(raw)}, indent=2), encoding="utf-8")
        return key


@dataclass(frozen=True)
class NexusIdentity:
    """The user's NEXUS: every device paired into it shares this id and its memory."""

    nexus_id: str

    @classmethod
    def load_or_create(cls, path: Path) -> NexusIdentity:
        if path.exists():
            return cls(json.loads(path.read_text(encoding="utf-8"))["nexus_id"])
        identity = cls("nx-" + uuid.uuid4().hex)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"nexus_id": identity.nexus_id}, indent=2), encoding="utf-8")
        return identity
