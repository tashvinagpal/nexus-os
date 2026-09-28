"""Transcript cleaning and Devanagari/Gurmukhi -> romanized Hinglish/Punjabi."""

from __future__ import annotations

import pytest

from nexus.ai.conversation.responder import parse_confirmation
from nexus.ai.voice.text import clean, likely_invented, romanize


@pytest.mark.parametrize("hindi, latin", [
    ("चला दे", "chala de"),
    ("क्रोम खोलो", "krom kholo"),
    ("कितने बजे हैं", "kitne baje hain"),
    ("हाँ", "han"),
    ("नहीं", "nahin"),
    ("ठीक है", "thik hai"),
    ("बंद करो", "band karo"),
    ("करना", "karna"),          # medial inherent 'a' dropped
    ("समझना", "samajhna"),
    ("बदल", "badal"),          # ...but not in two neighbouring syllables
    ("मित्र", "mitra"),         # kept after a cluster ending in r
    ("शब्द", "shabd"),
    ("डाउनलोड्स फ़ोल्डर खोल दो", "daunlods foldar khol do"),
    ("याद रखो कि मेरा पसंदीदा गेम रॉकेट लीग है", "yad rakho ki mera pasandida gem roket lig hai"),
    ("कल मिलते हैं", "kal milte hain"),
    ("१२३", "123"),
])
def test_devanagari(hindi, latin):
    assert romanize(hindi) == latin


@pytest.mark.parametrize("punjabi, latin", [
    ("ਕਿੰਨੇ ਵਜੇ ਨੇ", "kinne vaje ne"),
    ("ਹਾਂ ਜੀ", "han ji"),
    ("ਚਲਾ ਦੇ", "chala de"),
    ("ਗੱਡੀ", "gaddi"),           # addak doubles the next consonant
    ("ਕ੍ਰੋਮ ਖੋਲ੍ਹ ਦਿਓ", "krom khol dio"),  # pairin haha (tone) left out
])
def test_gurmukhi(punjabi, latin):
    assert romanize(punjabi) == latin


def test_latin_and_mixed_text():
    assert romanize("open chrome") == "open chrome"
    assert romanize("GTA चला दे bhai") == "GTA chala de bhai"
    assert romanize("Chrome खोल दो।") == "Chrome khol do."


@pytest.mark.parametrize("spoken", ["हाँ", "हाँ कर दो", "ठीक है", "ਹਾਂ ਜੀ"])
def test_romanized_yes_is_still_a_yes(spoken):
    assert parse_confirmation(romanize(spoken)) is True


@pytest.mark.parametrize("spoken", ["नहीं", "मत करो", "ਨਹੀਂ"])
def test_romanized_no_is_still_a_no(spoken):
    assert parse_confirmation(romanize(spoken)) is False


def test_clean_removes_non_speech_markers():
    assert clean("[Music] open chrome (laughs) ♪") == "open chrome"
    assert clean("  *coughs*  ") == ""


@pytest.mark.parametrize("text, invented", [
    ("Thanks for watching!", True), ("Thank you.", True), ("Please subscribe to my channel", True),
    ("you", True), ("...", True),
    ("open chrome", False), ("thank you nexus, open chrome", False), ("what time is it", False),
])
def test_known_hallucinations(text, invented):
    assert likely_invented(text) is invented
