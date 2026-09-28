"""Preparing replies for speech: what is said, never what is shown."""

from __future__ import annotations

import pytest

from nexus.ai.voice.hinglish import to_devanagari_speech
from nexus.ai.voice.speech_text import clean_for_speech, prepare_for_speech


def said(text: str, **kw) -> str:
    return prepare_for_speech(text, **kw).text


@pytest.mark.parametrize("reply, spoken", [
    ("Sure! **Opening Chrome now.**\nhttps://example.com/some/long/path?x=1", "Sure! Opening Chrome now. example.com."),
    ("Opened [YouTube](https://youtube.com) for you.", "Opened YouTube for you."),
    ("Go to www.github.com/user/repo.", "Go to github.com."),
    ("# Result\n- CPU: 12%\n- RAM: 16 GB", "Result. CPU: 12%. RAM: 16 gigabytes."),
    ("Wow!!! That is great 🎉🔥", "Wow! That is great."),
    ("Saved to C:\\Users\\yug\\Downloads\\mods\\", "Saved to mods."),
    ("NEXUS says hi, e.g. hello & welcome.", "Nexus says hi, for example hello and welcome."),
    ("| Name | Size |\n|---|---|\n| GTA | 95 GB |", "Name, Size. GTA, 95 gigabytes."),
    ("GPU temperature is 45 C and it runs at 1.8 GHz.", "GPU temperature is 45 degrees Celsius and it runs at 1.8 gigahertz."),
    ("It's 3:42 PM.", "It's 3:42 PM."),
    ("> quoted *text* and _more_ ~~gone~~", "quoted text and more gone."),
])
def test_what_is_said(reply, spoken):
    assert said(reply) == spoken


def test_code_blocks_are_not_read_out():
    reply = "Here it is:\n```python\nimport os\nos.remove('x')\n```\nRun it with `python x.py`."
    assert said(reply) == "Here it is: The code is on the screen. Run it with python x.py."
    assert "import" not in said(reply)


def test_unfinished_code_block():
    assert said("Look:\n```\nprint(1)") == "Look: The code is on the screen."


@pytest.mark.parametrize("empty", ["", "   ", "***", "\U0001F389\U0001F525", "\n\n"])
def test_nothing_to_say(empty):
    assert prepare_for_speech(empty).chunks == ()


def test_the_reply_itself_is_never_changed():
    reply = "**Done.** See https://example.com"
    prepare_for_speech(reply)
    assert reply == "**Done.** See https://example.com"  # strings are immutable; the point is no shared state


def test_hindi_and_punjabi_text_pass_through():
    assert said("क्रोम खुल गया है। अब क्या करूँ?", language="hi") == "क्रोम खुल गया है। अब क्या करूँ?"
    assert said("ਮੈਂ ਹਾਲੇ ਕੁਝ ਨਹੀਂ ਕੀਤਾ।", language="pa") == "ਮੈਂ ਹਾਲੇ ਕੁਝ ਨਹੀਂ ਕੀਤਾ।"


def test_units_are_expanded_only_in_english():
    assert said("16 GB free", language="hinglish") == "16 GB free."


def test_long_replies_stop_after_whole_sentences():
    reply = " ".join(f"This is sentence number {i} of a long answer." for i in range(1, 40))
    speech = prepare_for_speech(reply, max_chars=300)
    assert speech.truncated
    assert speech.text.endswith("The rest is on the screen.")
    assert len(speech.text) < 380
    assert all(chunk.endswith(".") for chunk in speech.chunks)  # never cut mid-sentence


@pytest.mark.parametrize("language, tail", [("hi", "बाकी स्क्रीन पर है।"), ("hinglish", "Baaki screen par hai.")])
def test_rest_on_screen_in_the_reply_language(language, tail):
    reply = " ".join(["Yeh ek lamba jawab hai."] * 60)
    assert prepare_for_speech(reply, language=language, max_chars=100).chunks[-1].endswith(tail)


def test_chunks_follow_sentences_and_the_first_comes_alone():
    speech = prepare_for_speech("Opening Google Chrome. It should be ready in a second. Tell me if not.")
    assert speech.chunks[0] == "Opening Google Chrome."  # the first sound comes sooner
    assert all(len(c) <= 220 for c in speech.chunks)


def test_a_very_long_sentence_is_split_at_commas_then_words():
    sentence = ", ".join(f"item number {i}" for i in range(60)) + "."
    chunks = prepare_for_speech(sentence, max_chars=5000).chunks
    assert len(chunks) > 3 and all(len(c) <= 220 for c in chunks)
    assert " ".join(chunks).replace(" ", "") == sentence.replace(" ", "")
    words = prepare_for_speech("word " * 300, max_chars=5000).chunks
    assert all(len(c) <= 220 for c in words)


def test_clean_is_exposed_for_other_front_ends():
    assert clean_for_speech("**Hi** 🎉") == "Hi."


# --- Hinglish for the ear -----------------------------------------------------------------------

@pytest.mark.parametrize("hinglish, devanagari", [
    ("Haan bhai, ek second.", "हां भाई, एक second."),
    ("Ekdum badhiya bhai! Tu bata?", "एकदम बढ़िया भाई! तू बता?"),
    ("Maine abhi kuch kiya nahi hai.", "मैंने अभी कुछ किया नहीं है."),
    ("Kar doon?", "कर दूं?"),
    ("Kal subah milte hain.", "कल सुबह मिल्ते हैं."),  # rules: an approximation of the spelling
])
def test_romanized_hindi_becomes_devanagari_for_the_hindi_voice(hinglish, devanagari):
    assert to_devanagari_speech(hinglish) == devanagari


def test_english_words_names_and_acronyms_stay_english():
    assert to_devanagari_speech("GTA chala raha hoon, ek second.") == "GTA चला रहा हूं, एक second."
    assert to_devanagari_speech("Theek hai, Rocket League chala deta hoon.") == \
        "ठीक है, Rocket League चला देता हूं."
    assert to_devanagari_speech("Maine abhi check nahi kiya.") == "मैंने अभी check नहीं किया."
    assert to_devanagari_speech("valorant khol do", names=["valorant"]) == "valorant खोल दो"
    assert to_devanagari_speech("Main theek hoon, tum batao.") == "मैं ठीक हूं, तुम बताओ."


def test_nothing_is_translated_and_devanagari_is_left_alone():
    assert to_devanagari_speech("क्रोम खुल गया है।") == "क्रोम खुल गया है।"
    assert to_devanagari_speech("Do you want Chrome?") == "Do you want Chrome?"
