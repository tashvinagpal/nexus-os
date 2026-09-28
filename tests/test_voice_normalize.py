"""Transcript normalization: Hinglish, Hindi, Punjabi, Haryanvi -> what NEXUS understands.

Most inputs are the real transcripts from the first real-microphone test (2026-09-28):
what Whisper actually wrote when a real speaker said each phrase.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nexus.ai.conversation.responder import parse_confirmation
from nexus.ai.router import FastPathRouter
from nexus.ai.voice.normalize import Normalizer, sound_key

APPS = {  # display name -> names people use (as the real catalog generates them)
    "Grand Theft Auto V Legacy": {"gta", "gta v", "gta 5", "gtav", "grand theft auto v"},
    "Google Chrome": {"chrome", "google chrome"},
    "Steam": {"steam"},
    "OBS Studio": {"obs", "obs studio"},
    "Rocket League": {"rocket league"},
}


class FakeApps:
    def resolve(self, name: str, **_):
        low = " ".join(name.lower().split())
        for display, aliases in APPS.items():
            if low in aliases:
                return SimpleNamespace(confident=True, ambiguous=False,
                                       match=SimpleNamespace(display_name=display, aliases=frozenset(aliases)))
        return SimpleNamespace(confident=False, ambiguous=False, match=None)

    def catalog(self):
        return self


ROUTER = FastPathRouter(apps=FakeApps())


def understood(text: str) -> bool:
    return ROUTER.route(text) is not None or parse_confirmation(text) is not None


NORMALIZE = Normalizer(understood, lambda: [a for names in APPS.values() for a in names])


def action(text: str):
    plan = ROUTER.route(text)
    return (plan.actions[0].tool, plan.actions[0].args) if plan and plan.actions else None


# --- the real transcripts ---------------------------------------------------------------------

@pytest.mark.parametrize("heard, acted, tool, guessed", [
    # English: unchanged
    ("What time is it?", "What time is it?", "get_datetime", False),
    ("open my downloads folder", "open my downloads folder", "open_folder", False),
    # Hinglish "Chrome khol do": Whisper wrote English-looking words -> a guess (NEXUS asks first)
    ("Chrome, Coldo.", "Chrome khol do", "launch_app", True),
    # Hindi
    ("कितने बजे हैं?", "कितने बजे हैं?", "get_datetime", False),
    ("download folder खोलो", "download folder kholo", "open_folder", False),
    # Punjabi said like Hindi: Devanagari instead of Gurmukhi
    ("किन्ने वाजे, ने", "kinne vaje ne", "get_datetime", False),
    ("Chrome coldio", "Chrome khol do", "launch_app", True),
    # Haryanvi: "के टाइम हो रह्या सै" / "जीटीए चला दे भाई"
    ("के ताम हो रहा से", "kya time ho raha hai", "get_datetime", False),
    ("जी टी ए चलाडे बही", "gta chala de bhai", "launch_app", False),
])
def test_real_speaker_transcripts(heard, acted, tool, guessed):
    result = NORMALIZE(heard)
    assert result.text == acted
    assert result.guessed is guessed
    assert action(result.text)[0] == tool


def test_gta_said_with_a_garbled_wake_word_is_not_forced():
    # "Nexus, GTA chala de" -> "अग्स्छ जीटिय चलादे": "Nexus" became an unknown word. Unknown words are
    # never dropped to make a command fit ("don't open chrome" must never become "open chrome").
    result = NORMALIZE("अग्स्छ जीटिय चलादे")
    assert action(result.text) is None and not result.guessed
    assert result.text == "agschh jitiy chala de"  # the planner gets the exact text, no guesses
    fixed = NORMALIZE("जीटिय चलादे")  # without the garbled word, the near-miss "jitiy" is a guess
    assert fixed.text == "gta chala de" and fixed.guessed


@pytest.mark.parametrize("heard, answer, guessed", [
    ("अ, कर दो.", True, False),          # "haan, kar do" - the h dropped
    ("नहीं रहने दो", False, False),       # was not recognised as a no in the real test
    ("I'll do it.", None, False),          # a translation, not the speaker's words: not a yes
    ("Han, Cardu", True, True),            # English-looking yes: a guess -> asked again
    ("Nahee, Renny do.", False, True),     # a misheard no is still a no
    ("अन, कर्दू", True, False),
])
def test_spoken_answers(heard, answer, guessed):
    result = NORMALIZE(heard)
    assert parse_confirmation(result.text) is answer
    assert result.guessed is guessed


# --- general behaviour ---------------------------------------------------------------------------

@pytest.mark.parametrize("sentence", [
    "Open Crow.", "the car is cold", "I'm feeling cold, do something", "call mom", "what is GTA?",
    "I'm bored, what should I play?", "tell me a joke", "GTA ke baare mein batao",
    "don't open chrome", "Keetney Beach High",
])
def test_ordinary_sentences_are_not_rewritten(sentence):
    result = NORMALIZE(sentence)
    assert result.text == sentence and not result.guessed and result.changes == ()


def test_english_words_that_sound_like_commands_are_not_forced():
    assert NORMALIZE("I'm cold do something").text == "I'm cold do something"


@pytest.mark.parametrize("heard, acted", [
    ("क्रोम खोलो", "chrome kholo"),           # an English name in Devanagari is the same word
    ("मेरा क्रोम खोल दे", "mera chrome khol de"),
    ("स्टीम बंद करो", "steam band karo"),
    ("ओ बी एस चला दे", "obs chala de"),        # letters spelled out
    ("नेक्सस क्रोम खोलो", "chrome kholo"),     # wake word dropped
])
def test_devanagari_spellings_are_not_guesses(heard, acted):
    result = NORMALIZE(heard)
    assert result.text == acted and not result.guessed and action(result.text) is not None


@pytest.mark.parametrize("heard, acted", [
    ("ਕੀ ਟਾਈਮ ਹੋਇਆ", "kya time hua"),           # Punjabi in Gurmukhi
    ("कितणे बजे सै", "kitne baje sai"),          # Haryanvi, understood as-is by the fast path
    ("steam bandkaro", "steam band karo"),       # fused words split
    ("Rocket League chalade", "Rocket League chalade"),   # the fast path already knows the fused form
])
def test_regional_and_fused_forms(heard, acted):
    result = NORMALIZE(heard)
    assert result.text == acted and action(result.text) is not None


def test_regional_words_reach_the_planner_in_hinglish_without_guesses():
    result = NORMALIZE("mhara dost kal aavega, dasso")  # nothing the fast path does
    assert result.text == "mera dost kal aavega batao" and not result.guessed


def test_guesses_are_only_used_when_the_fast_path_then_understands_everything():
    result = NORMALIZE("coldo")  # "khol do" alone isn't a command
    assert result.text == "coldo" and not result.guessed


def test_no_guessing_on_long_sentences():
    long = "bhai mujhe lagta hai ki Chrome coldio karna chahiye shayad aaj"
    assert not NORMALIZE(long).guessed


def test_without_a_fast_path_only_exact_rules_apply():
    plain = Normalizer()
    assert plain("Chrome, Coldo.").text == "Chrome, Coldo."
    assert plain("के ताम हो रहा से").text == "ke tam ho raha se"


def test_sound_keys():
    assert sound_key("chrome") == sound_key("krom")
    assert sound_key("coldo") == sound_key("kholdo")
    assert sound_key("steam") == sound_key("stim")
    assert sound_key("kardu") == sound_key("kardo")
    assert sound_key("chala") != sound_key("kala")  # Hinglish ch is not k
