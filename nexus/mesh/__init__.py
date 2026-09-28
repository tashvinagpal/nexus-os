"""NEXUS mesh: your devices (PC, laptop, phone) working as one NEXUS.

v0.1 defines identities, the signed message format, the device registry and
the gateway that feeds authenticated messages into the conversation engine
and Action Kernel. Only an in-process transport exists; LAN discovery,
pairing, encrypted transport and an optional relay come later and plug in
behind the Transport interface. Nothing here requires a cloud service.
"""
