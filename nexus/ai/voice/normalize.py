"""Spoken Hinglish, Hindi, Punjabi and Haryanvi -> text the rest of NEXUS understands.

Whisper writes what it hears in whichever script it picked: Devanagari for
Hindi (and often for Punjabi and Haryanvi, which it labels Hindi), Gurmukhi,
or English letters with English-looking guesses ("khol do" -> "Coldo"). The
fast path and the planner were built and tested on romanized Hinglish, so the
transcript is brought to that form in steps, from exact to guessing:

  1. script     Devanagari/Gurmukhi -> Latin letters                   exact
                (kept as written if the fast path already understands it)
  2. dialect    fused and regional forms -> common Hinglish             exact rules
                "chalade" -> "chala de", Punjabi "dio" -> "do", "kinne vaje" -> "kitne baje",
                Haryanvi/Punjabi questions "ke/ki ... sai" -> "kya ... hai"
  3. sounds     words that sound like a command word                    see below
                "coldo" -> "khol do", "tam" -> "time", "bahi" -> "bhai"
  4. names      words that sound like an installed app's name           see below
                "क्रोम" -> "chrome", "जी टी ए" / "जीटिय" -> "gta"

Steps 3-4 only ever run on short utterances, and their result is used only if
the fast path then understands the whole request; otherwise the planner gets
the exact text from steps 1-2. An English name written in Devanagari that
sounds exactly like the name ("क्रोम") is a spelling, not a guess. Everything
else from steps 3-4 - an English-looking mishearing ("Coldo"), or a word one
sound away ("जीटिय") - is a GUESS: it is reported as such and the conversation
engine asks before acting on it. Nothing here decides permission.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable

from nexus.ai.conversation.engine import strip_wake_word
from nexus.ai.voice.text import romanize

_INDIC = re.compile(r"[ऀ-ॿ਀-੿]")
_SPLIT = re.compile(r"[\s,;!?।॥\"“”]+")
_MAX_GUESS_WORDS = 7  # only short, command-like utterances are ever guessed at


@dataclass(frozen=True)
class Understood:
    text: str                       # what NEXUS acts on
    changes: tuple[str, ...] = ()   # what was changed and why, e.g. '"coldo" -> "khol do" (sounds alike)'
    guessed: bool = False           # words were replaced by similar-sounding ones: ask before acting


@dataclass(frozen=True)
class _Word:
    text: str
    indic: bool = False  # written in Devanagari/Gurmukhi by the speech model


# --- sounds --------------------------------------------------------------------------------

_SOUND_RULES = (("chr", "kr"), ("chh", "C"), ("ch", "C"),  # English "chrome" is k; Hinglish "chala" is ch
                ("sh", "S"), ("ph", "f"), ("ck", "k"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"),
                ("bh", "b"), ("jh", "j"), ("q", "k"), ("x", "ks"), ("w", "v"), ("z", "j"), ("y", "i"),
                ("ea", "i"), ("ee", "i"), ("oo", "u"), ("aa", "a"),  # English "steam" = "stim"
                ("u", "o"))  # "do"/"du", "khol"/"khul": the vowel varies, the word doesn't


def sound_key(word: str) -> str:
    """How a word sounds, roughly, whether spelled the English or the Hinglish way."""
    w = re.sub(r"[^a-z]", "", word.lower())
    for a, b in _SOUND_RULES:
        w = w.replace(a, b)
    w = re.sub(r"c(?=[eiy])", "s", w).replace("c", "k")
    w = re.sub(r"(?<=[^aeiou])h", "", w)            # leftover aspiration: "khol" = "kol"
    if len(w) > 3 and w.endswith("e") and w[-2] not in "aeiou":
        w = w[:-1]                                   # English silent e: "chrome" = "krom", "time" = "tim"
    return re.sub(r"(.)\1+", r"\1", w)               # "kholl", "gaddi" -> single letters


def _distance(a: str, b: str) -> int:
    if abs(len(a) - len(b)) > 1:
        return 2
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _close(key: str, target: str) -> int | None:
    """0 for the same sound, 1 for one sound off (longer words only), else None."""
    if not key or not target:
        return None
    if key == target:
        return 0
    if max(len(key), len(target)) >= 5 and _distance(key, target) == 1:
        return 1
    if len(key) >= 3 and key.replace("h", "") == target.replace("h", ""):
        return 1  # the h of an aspirated sound written elsewhere: "bahi" = "bhai", "kohl" = "khol"
    return None


# Command words people say to NEXUS, in the romanized Hinglish the fast path knows. Multi-word
# entries also match when spoken as one word ("kholdo").
COMMAND_WORDS = ("khol do", "khol de", "kholo", "khol", "chala de", "chala do", "chalao", "chala", "chalu kar",
                 "band kar do", "band karo", "band kar", "kar do", "kar de", "karo", "bhai", "yaar", "please",
                 "time", "baje", "kitne", "kya", "hai", "hua", "raha", "folder", "downloads", "documents",
                 "desktop", "open", "close", "start", "launch",
                 # answers to "should I...?" (a guessed yes is asked again; a no always cancels)
                 "haan", "nahi", "rehne do", "chhod do", "theek hai")
_FILLERS = {"bhai", "yaar", "please"}  # correcting these never changes what is done
_SPOKEN_VARIANTS = {"taim": "time", "tam": "time", "tem": "time", "teim": "time", "tym": "time"}

# How English letters are said, as Hinglish speakers write them ("GTA" = "ji ti e").
_LETTERS = {"a": "e", "b": "bi", "c": "si", "d": "di", "e": "i", "f": "ef", "g": "ji", "h": "ech", "i": "ai",
            "j": "je", "k": "ke", "l": "el", "m": "em", "n": "en", "o": "o", "p": "pi", "q": "kyu", "r": "ar",
            "s": "es", "t": "ti", "u": "yu", "v": "vi", "w": "dablyu", "x": "eks", "y": "vai", "z": "jed"}


# --- dialect (exact rules) --------------------------------------------------------------------

_FUSED = re.compile(r"(khol|chala|chalu|band|kar|rehne|rahne|chhod|chod|laga|bata|dikha)"
                    r"(do|de|dio|diyo|dijiye|dena|karo|kardo|karde)")
_WORDS = {  # regional or alternative forms -> the Hinglish form
    "dio": "do", "diyo": "do", "deo": "do", "dyo": "do",          # Punjabi "kar dio", "khol dio"
    "kinne": "kitne", "vaje": "baje", "vajje": "baje", "bajje": "baje", "baaje": "baje",  # Punjabi time
    "hoya": "hua", "hoiya": "hua", "hoeya": "hua", "hoia": "hua", "hoiaa": "hua",  # "happened"
    "rahya": "raha", "rahiya": "raha", "reha": "raha", "rya": "raha", "ria": "raha",
    "koni": "nahi",                                                # Haryanvi "not"
    "mhara": "mera", "mhari": "meri", "mhare": "mere",            # Haryanvi "my"
    "dasso": "batao", "daso": "batao", "dass": "bata",             # Punjabi "tell"
}
_QUESTION_START = {"ke", "ki", "kai"}   # Haryanvi/Punjabi "what" at the start of a question
_COPULA_END = {"sai", "sae", "se"}     # Haryanvi "hai" at the end


def _tokens(text: str) -> list[_Word]:
    """Words, each romanized on its own so it is known which ones were written in Indic script."""
    words = []
    for token in _SPLIT.split(text.strip()):
        token = token.strip(".") if not re.search(r"\w\.\w", token) else token  # keep "youtube.com"
        if not token:
            continue
        if _INDIC.search(token):
            words.extend(_Word(part, True) for part in romanize(token).split())
        else:
            words.append(_Word(token))
    return words


def _text(words: list[_Word]) -> str:
    return " ".join(w.text for w in words)


def _dialect(words: list[_Word]) -> tuple[list[_Word], list[str]]:
    out, changes = [], []
    for word in words:
        low = word.text.lower()
        fused = _FUSED.fullmatch(low)
        if fused:
            tail = fused.group(2)
            tail_parts = ["kar", tail[3:]] if tail in ("kardo", "karde") else [tail]
            parts = [fused.group(1), *(_WORDS.get(p, p) for p in tail_parts)]
            changes.append(f'"{word.text}" -> "{" ".join(parts)}"')
            out.extend(_Word(p, word.indic) for p in parts)
        elif low in _WORDS:
            changes.append(f'"{word.text}" -> "{_WORDS[low]}"')
            out.append(_Word(_WORDS[low], word.indic))
        else:
            out.append(word)
    return out, changes


def _question_forms(words: list[_Word]) -> tuple[list[_Word], list[str]]:
    """Haryanvi/Punjabi question shape: "ke tam ho raha se" -> "kya ... hai" (tried, never forced)."""
    out, changes = list(words), []
    if len(out) >= 3 and out[0].text.lower() in _QUESTION_START:
        changes.append(f'"{out[0].text}" -> "kya"')
        out[0] = _Word("kya", out[0].indic)
    if len(out) >= 3 and out[-1].text.lower() in _COPULA_END:
        changes.append(f'"{out[-1].text}" -> "hai"')
        out[-1] = _Word("hai", out[-1].indic)
    return out, changes


# --- sound-alike replacements -------------------------------------------------------------------

@dataclass(frozen=True)
class _Target:
    text: str          # what replaces the sound-alike words
    spoken_words: int  # how many words it is said as ("khol do" = 2, "gta" said as letters = 3)
    letters: bool = False


def _replace(words: list[_Word], targets: dict[str, _Target], *, skip: set[str],
             max_n: int) -> tuple[list[_Word], list[str], bool]:
    """Replace runs of words that sound like a target. Returns (words, changes, guessed).

    A run matches only word for word, or as one fused word ("kholdo"); a name spelled out
    letter by letter ("ji ti e") must match exactly. Two different, equally close targets
    mean no change at all.
    """
    out: list[_Word] = []
    changes: list[str] = []
    guessed, i = False, 0
    while i < len(words):
        best = None
        for n in range(max_n, 0, -1):
            window = words[i:i + n]
            if len(window) < n or any(w.text.lower() in skip for w in window):
                continue
            key = sound_key("".join(w.text for w in window))
            hits = []
            for tkey, target in targets.items():
                if n != 1 and n != target.spoken_words:
                    continue
                d = _close(key, tkey)
                if d is None or (d and n > 1 and target.letters):
                    continue
                hits.append((d, target.text))
            hits.sort()
            if not hits or (len(hits) > 1 and hits[0][0] == hits[1][0] and hits[0][1] != hits[1][1]):
                continue  # nothing close, or two different words equally close: don't guess
            distance, text = hits[0]
            best = (n, distance, None if _text(window).lower() == text.lower() else text)
            break
        if best is None:
            out.append(words[i])
            i += 1
            continue
        n, distance, text = best
        window = words[i:i + n]
        i += n
        if text is None:  # already right
            out.extend(window)
            continue
        spelling = distance == 0 and all(w.indic for w in window)  # "क्रोम" is simply "chrome" in Devanagari
        guess = not spelling and text not in _FILLERS
        guessed |= guess
        changes.append(f'"{_text(window)}" -> "{text}" ({"sounds alike" if guess else "same word"})')
        out.extend(_Word(t) for t in text.split())
    return out, changes, guessed


_COMMAND_KEYS = {sound_key(w.replace(" ", "")): _Target(w, len(w.split())) for w in COMMAND_WORDS}
_COMMAND_KEYS.update({sound_key(v): _Target(w, 1) for v, w in _SPOKEN_VARIANTS.items()})
_COMMAND_TOKENS = {t for w in COMMAND_WORDS for t in w.split()}


def _name_keys(names: Iterable[str]) -> dict[str, _Target]:
    keys: dict[str, _Target] = {}
    for name in names:
        name = name.strip().lower()
        if not name or name in _COMMAND_TOKENS:
            continue
        keys.setdefault(sound_key(name.replace(" ", "")), _Target(name, len(name.split())))
        if re.fullmatch(r"[a-z]{2,5}", name):  # could be said letter by letter: "gta" = "ji ti e"
            keys.setdefault(sound_key("".join(_LETTERS[c] for c in name)), _Target(name, len(name), letters=True))
    return {k: v for k, v in keys.items() if len(k) >= 3}


class Normalizer:
    """Brings a transcript to the form NEXUS understands (see the module docstring)."""

    def __init__(self, understood: Callable[[str], bool] | None = None,
                 names: Callable[[], Iterable[str]] | None = None) -> None:
        self._understood = understood
        self._names = names
        self._name_cache: tuple[int, dict[str, _Target]] | None = None

    def _ok(self, text: str) -> bool:
        return self._understood is not None and bool(text) and self._understood(strip_wake_word(text))

    def _name_targets(self) -> dict[str, _Target]:
        if self._names is None:
            return {}
        names = sorted(set(self._names()))
        stamp = hash(tuple(names))
        if self._name_cache is None or self._name_cache[0] != stamp:
            self._name_cache = (stamp, _name_keys(names))
        return self._name_cache[1]

    def __call__(self, text: str) -> Understood:
        text = text.strip()
        indic = bool(_INDIC.search(text))
        if indic:
            text = strip_wake_word(text)
        if self._ok(text):
            return Understood(text)  # understood as written (the fast path knows some Devanagari too)
        words = _tokens(text)
        changes = ["romanized"] if indic else []
        base = _text(words)
        if self._ok(base):
            return Understood(base, tuple(changes))

        exact, found = _dialect(words)
        changes += found
        if self._ok(_text(exact)):
            return Understood(_text(exact), tuple(changes))
        question, found = _question_forms(exact)
        if found and self._ok(_text(question)):
            return Understood(_text(question), tuple(changes + found))
        # The planner gets the exact text only - no guesses (and English exactly as transcribed).
        planner_text = Understood(_text(exact) if changes else text, tuple(changes))

        if self._understood is None or len(exact) > _MAX_GUESS_WORDS:
            return planner_text
        changes += found
        sounds, found, guessed = _replace(question, _COMMAND_KEYS, skip=set(), max_n=2)
        changes += found
        if found and self._ok(_text(sounds)):
            return Understood(_text(sounds), tuple(changes), guessed)
        named, found, named_guess = _replace(sounds, self._name_targets(), skip=_COMMAND_TOKENS, max_n=3)
        if found and self._ok(_text(named)):
            return Understood(_text(named), tuple(changes + found), guessed or named_guess)
        return planner_text
