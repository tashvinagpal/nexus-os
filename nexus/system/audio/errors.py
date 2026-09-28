"""Microphone problems. str(error) is written for the user; NEXUS keeps running after any of them."""

from __future__ import annotations


class AudioInputError(Exception):
    pass


class NoMicrophone(AudioInputError):
    """No usable microphone is connected (or every one is disabled/unplugged)."""


class MicrophoneDenied(AudioInputError):
    """The OS privacy setting blocks microphone access for desktop apps."""


class MicrophoneUnavailable(AudioInputError):
    """This microphone couldn't be opened (in use exclusively, format refused, driver error)."""


class MicrophoneLost(AudioInputError):
    """The microphone stopped delivering audio (unplugged, Bluetooth dropped)."""
