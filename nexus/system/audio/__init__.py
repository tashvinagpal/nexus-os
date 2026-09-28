"""Audio input: which microphone NEXUS listens with, and capturing from it.

Platform-independent. The OS-specific part is only the device list
(`Platform.audio_inputs()`, Windows Core Audio today) and PortAudio's host
API (WASAPI on Windows, ALSA/PulseAudio/PipeWire on Linux).
"""

from nexus.system.audio.capture import CaptureBackend, CaptureStream, PortAudioCapture
from nexus.system.audio.devices import MicChange, MicrophoneManager, MicScan, classify
from nexus.system.audio.errors import (AudioInputError, MicrophoneDenied, MicrophoneLost, MicrophoneUnavailable,
                                       NoMicrophone)

__all__ = ["AudioInputError", "CaptureBackend", "CaptureStream", "MicChange", "MicScan", "MicrophoneDenied",
           "MicrophoneLost", "MicrophoneManager", "MicrophoneUnavailable", "NoMicrophone", "PortAudioCapture",
           "classify"]
