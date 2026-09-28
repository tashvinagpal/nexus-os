"""From a written reply to what should be *said* - the reply itself is never changed.

The screen and the logs keep the original text. Speech gets a prepared copy:

  - markdown formatting, headings, bullets and tables are not read out
  - code blocks become "The code is on the screen."
  - links become just their site ("github.com"), never read letter by letter;
    markdown links keep their words
  - file paths become their last part ("C:\\Users\\me\\Downloads\\mods" -> "mods")
  - emojis and decorative symbols are dropped; "!!!" -> "!"
  - a few safe abbreviations are expanded ("e.g.", units after a number: "16 GB")
  - long replies stop after a few sentences: "The rest is on the screen."
  - the text is cut into sentence-sized chunks, never mid-sentence unless a
    single sentence is too long (then at a comma or a space)

Nothing is translated: Hindi stays Hindi, Hinglish stays Hinglish.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_REST_ON_SCREEN = {"en": "The rest is on the screen.", "hi": "बाकी स्क्रीन पर है।",
                   "hinglish": "Baaki screen par hai.", "pa": "Baaki screen te hai."}
_CODE_ON_SCREEN = {"en": "The code is on the screen.", "hi": "कोड स्क्रीन पर है।",
                   "hinglish": "Code screen par hai.", "pa": "Code screen te hai."}

_CODE_BLOCK = re.compile(r"```.*?(?:```|$)", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>()\"']+", re.IGNORECASE)
_PATH = re.compile(r"(?:\b[A-Za-z]:\\|\\\\|~[\\/])(?:[^\s\\/:*?\"<>|]+[\\/])*([^\s\\/:*?\"<>|]+)[\\/]?")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d{1,2}[.)])\s+", re.MULTILINE)
_QUOTE = re.compile(r"^\s*>\s?", re.MULTILINE)
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*\*|__|\*|_|~~)(?=\S)(.+?)(?<=\S)\1")
_REPEATED = re.compile(r"([!?.,;:])\1+")
_SENTENCE_END = re.compile(r"(?<=[.!?।॥…])\s+|\n+")

_ABBREVIATIONS = [
    (re.compile(r"\be\.g\.", re.IGNORECASE), "for example"),
    (re.compile(r"\bi\.e\.", re.IGNORECASE), "that is"),
    (re.compile(r"\betc\.", re.IGNORECASE), "et cetera"),
    (re.compile(r"\bvs\.?(?=\s)", re.IGNORECASE), "versus"),
    (re.compile(r"\bapprox\.", re.IGNORECASE), "approximately"),
    (re.compile(r"\s&\s"), " and "),
    (re.compile(r"\bNEXUS\b"), "Nexus"),  # not spelled out letter by letter
]
_UNITS = {"TB": "terabytes", "GB": "gigabytes", "MB": "megabytes", "KB": "kilobytes", "GHz": "gigahertz",
          "MHz": "megahertz", "ms": "milliseconds", "°C": "degrees Celsius", "° C": "degrees Celsius",
          "C": "degrees Celsius"}
_UNIT = re.compile(r"(\d(?:[\d,.]*\d)?)\s?(TB|GB|MB|KB|GHz|MHz|ms|°\s?C|C)\b")


@dataclass(frozen=True)
class SpeechText:
    chunks: tuple[str, ...]  # what to synthesize, in order
    truncated: bool = False  # the reply was longer; the end is on the screen only

    @property
    def text(self) -> str:
        return " ".join(self.chunks)


def _drop_symbols(text: str) -> str:
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ("So", "Cs", "Co") or ch in "\u200d\ufe0f":  # emoji, pictographs, joiners
            continue
        out.append(ch)
    return "".join(out)


def _unit(match: re.Match) -> str:
    number, unit = match.group(1), match.group(2).replace(" ", "")
    return f"{number} {_UNITS.get(unit, unit)}"


def clean_for_speech(text: str, language: str = "en") -> str:
    """The reply as it should be heard (see module docstring). May be empty."""
    code_note = _CODE_ON_SCREEN.get(language, _CODE_ON_SCREEN["en"])
    text = unicodedata.normalize("NFC", text or "")
    text = _CODE_BLOCK.sub(f"\n{code_note}\n", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _MD_LINK.sub(r"\1", text)

    def link(match: re.Match) -> str:
        url = match.group(0).rstrip(".,;:!?")
        host = re.sub(r"^(?:https?://)?(?:www\.)?", "", url, flags=re.IGNORECASE).split("/")[0]
        trailing = match.group(0)[len(url):]
        return (host or "a link") + trailing  # the site, never the path letter by letter

    text = _URL.sub(link, text)
    text = _PATH.sub(lambda m: m.group(1), text)
    text = _TABLE_RULE.sub("", text)
    text = "\n".join(line.strip().strip("|").replace(" | ", ", ").replace("|", ", ") if "|" in line else line
                     for line in text.splitlines())
    text = _HEADING.sub("", text)
    text = _BULLET.sub("", text)
    text = _QUOTE.sub("", text)
    for _ in range(2):  # nested emphasis
        text = _EMPHASIS.sub(r"\2", text)
    text = _drop_symbols(text)
    for pattern, spoken in _ABBREVIATIONS:
        text = pattern.sub(spoken, text)
    if language == "en":
        text = _UNIT.sub(_unit, text)
    text = _REPEATED.sub(lambda m: "..." if m.group(1) == "." and len(m.group(0)) >= 3 else m.group(1), text)
    text = re.sub(r"[#*_~^=<>{}\[\]\\]+", " ", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    # A line without an ending of its own still ends a sentence when spoken.
    lines = [line if not line or line[-1] in ".!?।॥…,:;" else line + "." for line in lines if line]
    return " ".join(lines).strip()


def _split_long(sentence: str, limit: int) -> list[str]:
    if len(sentence) <= limit:
        return [sentence]
    parts, current = [], ""
    for piece in re.split(r"(?<=[,;:—–])\s+", sentence):
        if len(piece) > limit:  # no comma to split at: split between words
            words = piece.split()
            piece, extra = "", []
            for word in words:
                if piece and len(piece) + 1 + len(word) > limit:
                    extra.append(piece)
                    piece = word
                else:
                    piece = f"{piece} {word}".strip()
            if current:
                parts.append(current)
                current = ""
            parts.extend(extra)
        if current and len(current) + 1 + len(piece) > limit:
            parts.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        parts.append(current)
    return parts


def prepare_for_speech(text: str, *, language: str = "en", max_chars: int = 600,
                       chunk_chars: int = 220) -> SpeechText:
    """Cleaned, shortened (whole sentences only) and chunked speech for one reply."""
    spoken = clean_for_speech(text, language)
    if not spoken:
        return SpeechText(())
    sentences = [s.strip() for s in _SENTENCE_END.split(spoken) if s and s.strip()]
    kept, used, truncated = [], 0, False
    for sentence in sentences:
        if kept and used + len(sentence) > max_chars:
            truncated = True
            break
        kept.append(sentence)
        used += len(sentence) + 1
    if truncated:
        kept.append(_REST_ON_SCREEN.get(language, _REST_ON_SCREEN["en"]))
    chunks: list[str] = []
    current = ""
    for sentence in kept:
        for piece in _split_long(sentence, chunk_chars):
            if current and len(current) + 1 + len(piece) > chunk_chars:
                chunks.append(current)
                current = piece
            else:
                current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    # Say the first sentence on its own: the first sound comes sooner.
    if len(chunks) == 1 and len(kept) > 1 and len(kept[0]) < len(chunks[0]):
        chunks = [kept[0], chunks[0][len(kept[0]):].strip()]
    return SpeechText(tuple(c for c in chunks if c), truncated)
