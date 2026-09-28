# NEXUS mesh (cross-device)

Code: `nexus/security/identity.py`, `nexus/mesh/`. Status: identities,
message format, registry, gateway and an in-process transport are built and
tested. **No network transport, discovery or pairing exists yet.**

## Identities

- `NexusIdentity.nexus_id` - one per user; all their devices share it.
- `DeviceKey` - an Ed25519 key pair per device. `device_id = "nxd-" +
  base32(sha256(public_key))`, so a device can't claim another's id.
- Stored in `var/identity/` (protected from all tools). v0.1 keeps the private
  key in a file; OS-backed storage (DPAPI/TPM, Android Keystore, Secure
  Enclave) comes with the network layer.
- `DeviceRegistry` - paired devices (id, public key, name, role, revoked).
  Refuses records whose id doesn't match their key.

## Envelope

```json
{"v": 1, "msg_id": "...", "nexus_id": "nx-...", "sender": "nxd-phone", "recipient": "nxd-pc",
 "sent_at_ms": 1790000000000, "kind": "user_input", "body": {"text": "start GTA on my PC"},
 "sig": "<Ed25519 over canonical JSON of everything except sig>"}
```

Kinds: `user_input`, `action_request`, `confirm`, `status_request` (client ->
core); `turn_event`, `status`, `error` (core -> client). Bodies are validated
per kind (no unknown fields); envelopes over 64 KB are rejected.

## Gateway checks (in order)

1. protocol version, 2. same NEXUS, 3. addressed to this device, 4. sender is
paired and not revoked, 5. signature valid, 6. timestamp within 120 s,
7. `msg_id` not seen before (replay). Failures are audited and get **no reply**.

Then the request goes to the same ConversationEngine / Action Kernel a local
user uses, tagged `device=<sender>`:

- anything not in `[remote] allow_without_confirmation` (default:
  `get_system_info`, `get_datetime`) **needs confirmation**, even inside safe folders;
- **per-device permissions** in `policy.toml` on the PC:
  `[remote.devices."nxd-..."]` with its own `allow_without_confirmation` and
  `blocked` - pairing alone grants nothing beyond the defaults;
- `[remote] blocked` tools are refused from every other device;
- protected paths, HIGH risk etc. apply exactly as locally;
- private memories are never returned to another device;
- the confirmation can be answered by **that phone or on the PC** - never by a
  third device.

## Client side (`nexus/mesh/client.py`)

`MeshClient` is what the phone (or laptop) app is built on: `say()`,
`request()`, `confirm()`, `status()`. It runs no tools and has no authority; it
pins the core's public key from pairing and drops anything else. `status()`
includes the core's capabilities (RAM, GPU, VRAM, supported models, battery,
network), so the phone knows what the PC can run that it can't.

## Not built yet (planned, behind the `Transport` interface)

LAN discovery (mDNS), pairing (short code/QR shown on both screens), an
encrypted channel (Noise or mutual TLS with pinned keys), optional relay with
end-to-end encryption, phone app, memory sync. Signatures already make
envelopes authentic through any transport; confidentiality will come from the
channel.
