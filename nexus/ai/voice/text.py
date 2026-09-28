"""From a raw transcript to the text NEXUS acts on.

1. Clean: drop Whisper's non-speech markers ("[Music]", "(laughs)", "♪").
2. Reject: phrases Whisper is known to invent from noise ("Thanks for watching!")
   when the model itself wasn't sure (they are kept if confidently heard).
3. Romanize: Whisper writes Hindi in Devanagari and Punjabi in Gurmukhi, but
   the fast path and the planner were built and evaluated on romanized
   Hinglish/Punjabi ("GTA chala de", "kinne vaje ne"). A deterministic
   transliteration turns "चला दे" into "chala de" so both scripts reach the
   same, tested code. Latin text passes through unchanged.
"""

from __future__ import annotations

import re
import unicodedata

_MARKERS = re.compile(r"\[[^\]]*\]|\([^)]*\)|[♪♫]+|\*[^*]*\*")
_HALLUCINATIONS = re.compile(
    r"^(?:thanks? (?:you )?(?:so much )?for watching|thank you(?: (?:very|so) much)?|"
    r"(?:please )?(?:like and )?subscribe(?: to (?:my|the) channel)?|subtitles? by .*|"
    r"you|bye|bye bye|okay|\.+|we'll be right back|see you (?:next time|in the next video))[.!? ]*$",
    re.IGNORECASE)


def clean(text: str) -> str:
    text = _MARKERS.sub(" ", unicodedata.normalize("NFC", text))
    return re.sub(r"\s+", " ", text).strip(" \t-")


def likely_invented(text: str) -> bool:
    """A phrase Whisper often produces from silence or noise."""
    return bool(_HALLUCINATIONS.match(text.strip()))


# --- transliteration (Devanagari; Gurmukhi is mapped onto it) ------------------------

_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "ळ": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    # ड़/ढ़ as Hinglish writers usually spell them: "chhod", "padh" (not "chhor", "parh")
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "d", "ढ़": "dh", "फ़": "f", "य़": "y",
}
_NUKTA_FORMS = {"क": "क़", "ख": "ख़", "ग": "ग़", "ज": "ज़", "ड": "ड़", "ढ": "ढ़", "फ": "फ़", "य": "य़"}
_VOWELS = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri", "ए": "e", "ऐ": "ai",
           "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e"}
_SIGNS = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
          "ॉ": "o", "ॅ": "e"}
_NASAL = {"ं": "n", "ँ": "n"}
_VIRAMA, _NUKTA, _VISARGA = "्", "़", "ः"
_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)} | {chr(0x0A66 + i): str(i) for i in range(10)}

# Gurmukhi letters that aren't simply Devanagari + 0x100.
_GURMUKHI_SPECIAL = {"ੰ": "ं", "ੱ": "ੱ", "ਂ": "ं", "ਖ਼": "ख़", "ਗ਼": "ग़", "ਜ਼": "ज़", "ੜ": "ड़", "ਫ਼": "फ़",
                     "ਲ਼": "ळ", "ਸ਼": "श", "ੲ": "", "ੳ": ""}


def _to_devanagari(text: str) -> str:
    out = []
    # Subjoined ha (pairin haha) marks tone in Punjabi; romanized Punjabi leaves it out ("ਖੋਲ੍ਹ" khol).
    for ch in unicodedata.normalize("NFC", text).replace("੍ਹ", ""):
        if ch in _GURMUKHI_SPECIAL:
            out.append(_GURMUKHI_SPECIAL[ch])
        elif "਀" <= ch <= "੿":
            out.append(chr(ord(ch) - 0x100))
        else:
            out.append(ch)
    return "".join(out)


def _is_indic(ch: str) -> bool:
    return "ऀ" <= ch <= "ॿ" or "਀" <= ch <= "੿"


def _word(word: str) -> str:
    """One Devanagari word -> Latin, dropping the inherent 'a' where Hindi/Punjabi speakers do."""
    units: list[list[str]] = []  # [consonant or vowel, vowel text, kind]
    double_next = False
    for ch in word:
        if ch == "ੱ":  # Gurmukhi addak: the next consonant is doubled
            double_next = True
            continue
        if ch == _NUKTA and units and units[-1][2] in ("schwa",) and units[-1][3] in _NUKTA_FORMS:
            base = _NUKTA_FORMS[units[-1][3]]
            units[-1][0] = _CONSONANTS[base]
            continue
        if ch in _CONSONANTS:
            sound = _CONSONANTS[ch]
            if double_next:
                sound = sound[0] + sound
                double_next = False
            units.append([sound, "a", "schwa", ch])
        elif ch in _SIGNS and units and units[-1][2] == "schwa":
            units[-1][1], units[-1][2] = _SIGNS[ch], "sign"
        elif ch == _VIRAMA and units and units[-1][2] == "schwa":
            units[-1][1], units[-1][2] = "", "virama"
        elif ch in _VOWELS:
            units.append(["", _VOWELS[ch], "vowel", ch])
        elif ch in _NASAL:
            units.append(["", "n", "nasal", ch])
        elif ch == _VISARGA:
            units.append(["", "h", "nasal", ch])
        elif ch in _DIGITS:
            units.append(["", _DIGITS[ch], "other", ch])
        elif ch in ("।", "॥"):
            units.append(["", ".", "other", ch])
        else:
            units.append(["", ch, "other", ch])

    consonants = [i for i, u in enumerate(units) if u[0]]
    # Word-final inherent 'a' is silent ("खोल" khol, "डाउनलोड्स" downloads, "एस" es) except after a
    # cluster ending in r/y/v ("मित्र" mitra) and in one-letter words ("न" na).
    if len(consonants) > 1 or (consonants and consonants[-1] > 0):
        last = consonants[-1]
        after_cluster = last > 0 and units[last - 1][2] == "virama"
        if (units[last][2] == "schwa" and all(u[2] == "nasal" for u in units[last + 1:])
                and not (after_cluster and units[last][3] in "रयव")):
            units[last][1] = ""
    # Medial: V C(a) C V -> drop the 'a' ("करना" karna, "कितने" kitne, "डाउनलोड" daunlod),
    # right to left, so two neighbouring syllables never both lose it.
    for i in range(len(units) - 2, 0, -1):
        unit, prev, nxt = units[i], units[i - 1], units[i + 1]
        if unit[2] != "schwa" or not unit[1]:
            continue
        if nxt[0] and nxt[1] and prev[1] and prev[2] in ("schwa", "sign", "vowel"):
            unit[1] = ""
    return "".join(u[0] + u[1] for u in units)


def romanize(text: str) -> str:
    """Devanagari/Gurmukhi -> romanized Hinglish/Punjabi; anything else unchanged."""
    if not any(_is_indic(ch) for ch in text):
        return text
    text = _to_devanagari(text)
    parts = re.split(r"(\s+|[,.!?;:\"'()\-]+)", text)
    out = "".join(_word(p) if any(_is_indic(ch) for ch in p) else p for p in parts)
    return out.replace("।", ".").strip()
