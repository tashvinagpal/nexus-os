"""Recording one utterance: speech start/end, silence, never-forever limits, cancellation, device loss."""

from __future__ import annotations

import threading

import pytest

from nexus.ai.voice.base import ListeningCancelled, NoSpeech
from nexus.ai.voice.recorder import EnergyVAD, Recorder, VADConfig, level
from nexus.system.audio.errors import MicrophoneLost
from tests.voice_fixtures import RATE, FakeStream, TickClock, mix, noise, silence, voice

CFG = VADConfig()


def seconds(pcm: bytes, rate: int = RATE) -> float:
    return len(pcm) / 2 / rate


def record(audio: bytes, cfg: VADConfig = CFG, **kw):
    return Recorder(cfg).record(FakeStream(audio, **kw))


def test_levels():
    assert level(silence(30)) == 0.0
    assert 50 < level(noise(30, 100)) < 150
    assert level(voice(300, 4000)) > 1500


def test_speech_then_pause_is_one_utterance():
    audio = silence(500) + voice(1500) + silence(1500)
    utt = record(audio)
    assert not utt.truncated
    assert 1.3 <= utt.speech_seconds <= 1.6
    # Speech + a little lead-in and tail, not the whole buffer.
    assert 1.6 <= seconds(utt.clip.pcm16) <= 2.8
    assert utt.clip.sample_rate == RATE


def test_start_of_speech_is_not_cut_off():
    audio = silence(600) + voice(1000) + silence(1200)
    utt = record(audio)
    # the clip starts before the voice does (pre-roll): its first 150 ms are the quiet lead-in
    assert level(utt.clip.pcm16[:RATE * 2 * 150 // 1000]) < 300


def test_speaking_right_away_during_noise_calibration_is_captured():
    audio = voice(1200) + silence(1200)
    utt = record(audio)
    assert utt.speech_seconds >= 0.4
    assert seconds(utt.clip.pcm16) >= 1.2               # all of it...
    assert utt.clip.pcm16[:960] == audio[:960]          # ...from the very first word


def test_silence_times_out_with_no_speech():
    cfg = VADConfig(no_speech_timeout_s=2.0)
    with pytest.raises(NoSpeech) as err:
        record(noise(5000, 80), cfg)
    assert "didn't hear anything in 2 seconds" in str(err.value)


def test_digital_silence_suggests_a_muted_microphone():
    cfg = VADConfig(no_speech_timeout_s=1.0)
    with pytest.raises(NoSpeech) as err:
        record(silence(3000), cfg)
    assert "muted" in str(err.value)


def test_never_records_forever_while_you_keep_talking():
    cfg = VADConfig(max_utterance_s=3.0)
    stream = FakeStream(b"", tail=voice(1000))  # endless speech
    utt = Recorder(cfg).record(stream)
    assert utt.truncated
    assert 3.0 <= seconds(utt.clip.pcm16) <= 3.6


def test_never_records_forever_with_endless_background_noise():
    cfg = VADConfig(no_speech_timeout_s=2.0)
    stream = FakeStream(b"", tail=noise(600, 150))
    with pytest.raises(NoSpeech):
        Recorder(cfg).record(stream)
    assert stream.reads < 200


def test_a_click_is_ignored_and_listening_continues():
    click = voice(150, 6000)
    audio = silence(600) + click + silence(1500) + voice(1000) + silence(1200)
    utt = record(audio)
    assert 0.8 <= utt.speech_seconds <= 1.3  # the later sentence, not the click


def test_noisy_room_speech_still_detected():
    room = noise(4000, 400)  # a fan: louder than the absolute floor
    speech = silence(800) + voice(1500, 5000) + silence(1700)
    utt = record(mix(speech, room))
    assert 1.2 <= utt.speech_seconds <= 1.8


def test_quiet_speaker_close_to_the_noise_is_not_mistaken_for_speech_by_noise_alone():
    with pytest.raises(NoSpeech):
        record(noise(8000, 400), VADConfig(no_speech_timeout_s=3.0))


def test_other_sample_rates_work():
    rate = 48000
    audio = silence(500, rate) + voice(1000, rate=rate) + silence(1200, rate)
    utt = Recorder(CFG).record(FakeStream(audio, rate=rate))
    assert utt.clip.sample_rate == rate and utt.speech_seconds >= 0.8


def test_cancel_stops_listening():
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(ListeningCancelled):
        Recorder(CFG).record(FakeStream(b"", tail=voice(1000)), cancel=cancel)


def test_cancel_from_another_thread():
    cancel = threading.Event()
    stream = FakeStream(b"", tail=noise(600, 50), realtime=True)
    threading.Timer(0.2, cancel.set).start()
    with pytest.raises(ListeningCancelled):
        Recorder(VADConfig(no_speech_timeout_s=30)).record(stream, cancel=cancel)


def test_device_that_stops_sending_audio_is_lost():
    stream = FakeStream(silence(300))  # then nothing at all
    with pytest.raises(MicrophoneLost) as err:
        Recorder(CFG, clock=TickClock(0.5)).record(stream)
    assert err.value.partial is False  # no speech yet: the caller may switch microphones silently


def test_device_lost_mid_sentence_is_flagged():
    stream = FakeStream(silence(400) + voice(800))  # unplugged while talking
    with pytest.raises(MicrophoneLost) as err:
        Recorder(CFG, clock=TickClock(0.5)).record(stream)
    assert err.value.partial is True


def test_speech_start_callback():
    started = []
    Recorder(CFG).record(FakeStream(silence(400) + voice(800) + silence(1200)),
                         on_speech=lambda: started.append(True))
    assert started == [True]


def test_vad_threshold_adapts_to_noise():
    vad = EnergyVAD(CFG)
    for _ in range(10):
        vad.voiced(500.0, speaking=False)
    assert vad.noise == 500.0
    assert vad.threshold() == 1500.0 and vad.threshold(speaking=True) == 1050.0
    assert vad.voiced(2000.0, speaking=False)
    assert not vad.voiced(900.0, speaking=False)


def test_calibration_during_speech_is_capped_and_a_loud_room_raises_it_later():
    vad = EnergyVAD(CFG)
    for _ in range(10):
        vad.voiced(2500.0, speaking=False)  # talking from the first moment
    assert vad.noise == CFG.max_start_noise
    for _ in range(60):
        vad.voiced(1000.0, speaking=False)  # a fan right next to the microphone
    assert 950 < vad.noise <= 1000 and not vad.voiced(1000.0, speaking=False)
