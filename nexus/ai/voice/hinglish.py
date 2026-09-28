"""Hinglish (and romanized Punjabi) for the ear: Latin-letter Hindi -> Devanagari, for speech only.

An English voice can't pronounce romanized Hindi ("Ekdum badhiya bhai" came out
as "Ectombatiabye" when checked by speech recognition), and a Hindi voice needs
Devanagari. So, for SPEECH ONLY - the reply on screen is never changed:

  - romanized Hindi words become Devanagari by spelling rules:
      "haan bhai ek second" -> "हां भाई एक second"
  - English words stay English (the Hindi voice says them the English way):
      a list of common English/tech words, installed app names, capitalized names
      in mid-sentence ("Chrome", "Rocket League"), and English-looking spellings
  - nothing is translated; only the script changes

The result is an approximation of how a Hinglish speaker would read it, not a
dictionary transliteration (romanized Hindi is ambiguous: "kal", "hai").
"""

from __future__ import annotations

import re
from typing import Iterable

# Words that stay English in a Hinglish sentence (lower case). Installed app names are added by the caller.
ENGLISH = frozenset("""
a an the and or but if so of in on at by for from with about into over after before
is are was were be been am do does did done have has had will would can could should may might must
i you he she it we they me him her us them my your his its our their this that these those
yes no ok okay sorry please thanks thank hello hey bye sure right good great nice cool
app apps game games folder folders file files second seconds minute minutes hour hours day days
screen pc computer laptop phone mobile system storage memory ram gpu cpu vram battery wifi internet online offline
download downloads upload update install uninstall open close start stop launch run running ready done error
settings setting music video videos song songs photo photos message messages chat mail email link website
browser chrome steam gta youtube google whatsapp instagram netflix spotify discord windows linux nexus
time date today tomorrow now check test voice text copy paste delete save restart shutdown volume
want need like know think make get go see look play use try find show tell help work new old all one two
what when where who why how which there here then than also just only very more most much many some any
""".split())
# The most common Hinglish words, spelled as Hindi writes them (the rules below cover the rest).
COMMON = {
    "haan": "हां", "han": "हां", "haa": "हां", "hanji": "हांजी", "haanji": "हांजी", "nahi": "नहीं", "nahin": "नहीं",
    "nai": "नहीं", "bhai": "भाई", "yaar": "यार", "ji": "जी", "kya": "क्या", "kyun": "क्यों", "kyu": "क्यों",
    "kaise": "कैसे", "kaisa": "कैसा", "kaisi": "कैसी", "hai": "है", "hain": "हैं", "ho": "हो", "hoon": "हूं",
    "hun": "हूं", "main": "मैं", "maine": "मैंने", "mein": "में", "mujhe": "मुझे", "mera": "मेरा", "meri": "मेरी",
    "mere": "मेरे", "tu": "तू", "tum": "तुम", "aap": "आप", "yeh": "यह", "ye": "ये", "woh": "वो", "wo": "वो",
    "kuch": "कुछ", "sab": "सब", "abhi": "अभी", "ek": "एक", "accha": "अच्छा", "acha": "अच्छा", "achha": "अच्छा",
    "theek": "ठीक", "thik": "ठीक", "badhiya": "बढ़िया", "ekdum": "एकदम", "bata": "बता", "batao": "बताओ",
    "samajh": "समझ", "sakta": "सकता", "sakti": "सकती", "sakte": "सकते", "raha": "रहा", "rahi": "रही", "rahe": "रहे",
    "gaya": "गया", "gayi": "गई", "kar": "कर", "karo": "करो", "karna": "करना", "kiya": "किया", "doon": "दूं",
    "chala": "चला", "chalo": "चलो", "khol": "खोल", "khul": "खुल", "band": "बंद", "baaki": "बाकी", "baki": "बाकी",
    "par": "पर", "pe": "पे", "se": "से", "ko": "को", "ka": "का", "ki": "की", "ke": "के", "aur": "और", "ya": "या",
    "bhi": "भी", "toh": "तो", "phir": "फिर", "fir": "फिर", "jaldi": "जल्दी", "thoda": "थोड़ा", "bahut": "बहुत",
    "bohot": "बहुत", "haje": "हाले", "keeta": "कीता", "dasso": "दस्सो", "tussi": "तुस्सी",
}
# Both English and Hindi words: Hindi right after a Hindi word ("khol do"), else English ("Do you...").
AMBIGUOUS = {"do": "दो", "to": "तो", "hi": "ही", "is": "इस", "us": "उस", "de": "दे", "the": "थे", "main": "मैं"}
_ENGLISH_SHAPES = re.compile(r"(?:tion|sion|ment|ness|ing|ed|ly|ph|ck|th$|ee[a-z]*s$|^wh|oo[a-z]*ed)$|[qxz]", re.I)

_CONSONANTS = [("chh", "छ"), ("ch", "च"), ("kh", "ख"), ("gh", "घ"), ("jh", "झ"), ("th", "थ"), ("dh", "ध"),
               ("ph", "फ"), ("bh", "भ"), ("sh", "श"), ("k", "क"), ("g", "ग"), ("j", "ज"), ("t", "त"), ("d", "द"),
               ("n", "न"), ("p", "प"), ("f", "फ़"), ("b", "ब"), ("m", "म"), ("y", "य"), ("r", "र"), ("l", "ल"),
               ("v", "व"), ("w", "व"), ("s", "स"), ("h", "ह"), ("z", "ज़"), ("q", "क़"), ("c", "क")]
# (latin, independent form, sign after a consonant)
_VOWELS = [("aa", "आ", "ा"), ("ai", "ऐ", "ै"), ("au", "औ", "ौ"), ("ee", "ई", "ी"), ("ii", "ई", "ी"),
           ("oo", "ऊ", "ू"), ("uu", "ऊ", "ू"), ("a", "अ", ""), ("i", "इ", "ि"), ("u", "उ", "ु"), ("e", "ए", "े"),
           ("o", "ओ", "ो")]
_LONG_FINAL = {"a": "ा", "i": "ी", "u": "ू"}  # word-final vowels in Hinglish are usually long: "chala", "nahi", "tu"
_VIRAMA = "्"


def _word_to_devanagari(word: str) -> str:
    w = word.lower()
    out: list[str] = []
    i, after_consonant = 0, False
    while i < len(w):
        vowel = next(((lat, ind, sign) for lat, ind, sign in _VOWELS if w.startswith(lat, i)), None)
        if vowel is not None:
            lat, ind, sign = vowel
            i += len(lat)
            final = i == len(w)
            # A nasal n at the end after a vowel: "haan" -> हां, "nahin" -> नहीं, "hain" -> हैं.
            nasal = w[i:] == "n" and (len(lat) == 2 or lat in ("i", "u") or (lat == "a" and w[i - 2:i] != "a"))
            if final and lat in _LONG_FINAL and after_consonant:
                sign = _LONG_FINAL[lat]
            elif final and lat in _LONG_FINAL and not after_consonant:
                ind = {"a": "आ", "i": "ई", "u": "ऊ"}[lat]
            out.append(sign if after_consonant else ind)
            if nasal:
                out.append("ं")
                i += 1
            after_consonant = False
            continue
        cons = next(((lat, dev) for lat, dev in _CONSONANTS if w.startswith(lat, i)), None)
        if cons is None:  # digits, apostrophes: keep
            out.append(w[i])
            i += 1
            after_consonant = False
            continue
        if after_consonant:
            out.append(_VIRAMA)  # two consonants in a row: a cluster ("kya" -> क्या)
        out.append(cons[1])
        i += len(cons[0])
        after_consonant = True
    return "".join(out)


def _is_english(word: str, index: int, english: frozenset[str]) -> bool:
    low = word.lower()
    if low in english or not low.isalpha():
        return True
    if index > 0 and word[:1].isupper():  # a name in mid-sentence: "Chrome", "Rocket"
        return True
    if word.isupper() and len(word) > 1:  # an acronym: GTA, PC
        return True
    return bool(_ENGLISH_SHAPES.search(low))


def to_devanagari_speech(text: str, *, names: Iterable[str] = ()) -> str:
    """Romanized Hindi words -> Devanagari; English words and everything else unchanged."""
    english = ENGLISH | {part for name in names for part in name.lower().split()}
    index = 0
    previous_hindi = False

    def convert(match: re.Match) -> str:
        nonlocal index, previous_hindi
        word = match.group(0)
        at, low = index, word.lower()
        index += 1
        if low in AMBIGUOUS and not (at == 0 and low == "main"):
            result = AMBIGUOUS[low] if previous_hindi else word
        elif low in COMMON:
            result = COMMON[low]
        elif _is_english(word, at, english):
            result = word
        else:
            result = _word_to_devanagari(word)
        previous_hindi = result != word
        return result

    # Sentence starts reset "mid-sentence" (a capital there is just a sentence start).
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for sentence in sentences:
        index, previous_hindi = 0, False
        out.append(re.sub(r"[A-Za-z']+", convert, sentence))
    return " ".join(out)
