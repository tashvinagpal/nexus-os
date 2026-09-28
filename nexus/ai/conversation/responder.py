"""Response generation: turns kernel outcomes into what NEXUS says.

- Turns with actions: the model's words are not used at all. NEXUS says a
  code-written acknowledgement in the user's style, the tool's own progress
  line, then the result from the Action Kernel's verified outcome.
- Conversation-only turns: the model's reply passes a claim check that
  replaces it if it says something was done, is being done, or asks its own
  "are you sure?" (only the kernel asks for confirmation).

The claim check is a safety net for conversation; the structural guarantee is
that action results are only ever worded from kernel outcomes.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping

from nexus.ai.conversation.types import Say, SayKind
from nexus.core.kernel import Outcome, Status
from nexus.tools.base import Tool

# Completion claims ("it's done") in English, Hinglish, Hindi and Punjabi. Allowed only as the
# result of an action the kernel verified.
_COMPLETION = [re.compile(p, re.IGNORECASE) for p in (
    r"\bi(?:'ve| have|'m| am)?\s+(?:just\s+|now\s+|already\s+)?(?:opened|launched|started|created|made|"
    r"deleted|closed|installed|moved|copied|sent|searched|changed|set|turned|played|ran|executed)\b",
    r"^\W*(?:done|all done|ok(?:ay)?,?\s+done)\b",
    r"\b(?:is|are|has been|have been)\s+(?:now\s+)?(?:open|opened|running|launched|started|created|"
    r"deleted|closed|ready)\b",
    r"\b(?:khol|chala|chalu|band|bana|kar|laga|start|delete|install|search|open)\s*(?:diya|diye|di|dia|dii)\b",
    r"\bho\s*(?:gaya|gya|gayi|gyi)\s*(?:hai)?\s*[.!]*\s*$",
    r"(?:कर|खोल|चला|बंद|बना|शुरू)\s*(?:दिया|दी|दिए)",
    r"हो\s*गया\s*(?:है)?\s*[।.!]*\s*$",
    r"(?:ਕਰ|ਖੋਲ੍ਹ|ਚਲਾ)\s*ਦਿੱਤਾ",
    r"\b(?:disabled|enabled|deleted|formatted|uninstalled|shut down|turned (?:off|on)|sent|opened|closed|"
    r"launched|started|stopped|restarted|rebooted)\b",
)]

# Activity claims ("doing it now", "one moment") and fake confirmations. True when an action is
# really running (NEXUS's own acknowledgement), false in a conversation-only turn.
_ACTIVITY = [re.compile(p, re.IGNORECASE) for p in (
    r"^\W*(?:(?:ok(?:ay)?|sure|haan|han|ji|bhai|yaar)[\s,!.]+){0,3}(?:now\s+)?(?:formatting|sending|deleting|"
    r"opening|launching|starting|closing|shutting|installing|uninstalling|disabling|enabling|creating|running|"
    r"playing|downloading|searching|checking|turning|restarting)\b",
    r"\b(?:please wait|one moment|just a (?:sec|second|moment)|hold on|ek (?:second|sec|minute|min|pal))\b",
    r"\b(?:kar|khol|chala|bhej|band|delete|install|start|search|dhundh)\s*(?:raha|rahi)\s*(?:hoon|hu|hun|hai)\b",
    # fake confirmations: only the Action Kernel asks
    r"\bare you sure\b",
    r"\b(?:pakka|sure)\s*(?:hai|ho)?\s*\?",
)]

# "I can't ... shut down your PC" / "I couldn't verify that Grand Theft Auto V Legacy started" are not
# claims: a negation earlier in the same clause covers the verb. The clause ends at punctuation, so
# "Don't worry, I've opened Chrome" still counts as a claim.
_NEGATED = re.compile(r"(?:\b(?:not|never|cannot|unable to|nahi|nahin|mat)|n't|\bcant|\bwont|\bdont)"
                      r"[^,.;:!?]*$", re.IGNORECASE)

# Hardware figures ("465 GB", "80%", "50 C") in a conversation-only reply were not measured in
# this turn. They are allowed only when the conversation already holds verified figures of the
# same kind (a follow-up about a real reading); otherwise they are likely invented.
_FIGURE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(gb|mb|tb|ghz|mhz|degrees?|°\s*c?|c|%)(?![a-z])", re.IGNORECASE)

_UNCHECKED = {
    "en": "I haven't measured that. Ask me something like \"check my storage\" and I'll read the real numbers.",
    "hinglish": "Maine abhi check nahi kiya. \"Storage check karo\" jaisa bolo toh asli numbers dekhta hoon.",
    "hi": "मैंने अभी जाँचा नहीं है। \"स्टोरेज चेक करो\" जैसा कहें तो असली आँकड़े देखता हूँ।",
    "pa": "Main haje check nahi keeta. \"Storage check karo\" kaho taan asli numbers dekhanga.",
}

_NOTHING_DONE = {
    "en": "I haven't done anything yet - I can't do that, or I misunderstood. Could you say it another way?",
    "hinglish": "Maine abhi kuch kiya nahi hai - ya toh main ye nahi kar sakta, ya samjha nahi. "
                "Thoda aur clear bologe?",
    "hi": "मैंने अभी कुछ नहीं किया है - या तो मैं यह नहीं कर सकता, या मैं समझा नहीं। दोबारा बताएँगे?",
    "pa": "Main haje kuch nahi keeta - ya taan main eh nahi kar sakda, ya samjheya nahi. Dubara dasso?",
}
_ASK = {"en": "Should I go ahead? (yes/no)", "hinglish": "Kar doon? (haan/nahi)",
        "hi": "क्या मैं आगे बढ़ूँ? (हाँ/नहीं)", "pa": "Kar dayan? (haan/nahi)"}

# --- confirmation answers (deterministic, never the language model) --------------

# Hindi, Hinglish, Punjabi and Haryanvi forms, in Latin letters (as typed, or romanized from speech)
# and in Devanagari/Gurmukhi. Saying no is always safe, so NO wins over YES in the same answer.
_FILLER = {"nexus", "bhai", "yaar", "please", "plz", "pls", "ji", "sir", "boss", "bro", "then", "toh", "it",
           "that", "ahead", "go", "do", "kar", "karo", "de", "hai", "dena", "dijiye", "chahiye", "chahie", "chaiye", "abhi",
           "na",  # Hinglish softener: "kar de na" means "please do it" (alone, "na" means no - see below)
           "a", "an", "uh", "um", "hmm",  # hesitation ("अ, कर दो": the model often drops the h of "haan")
           "dio", "diyo", "deo", "aa", "sai", "se", "thanks", "thank", "you",  # Punjabi / Haryanvi forms
           "है", "जी", "भाई", "करो", "कर", "दो", "दे", "चाहिए", "अभी", "सै", "अ", "अन", "ਜੀ", "ਹੈ", "ਕਰ", "ਦਿਓ", "ਦੇ", "ਆ"}
_YES = {"yes", "y", "yeah", "yep", "yup", "sure", "ok", "okay", "haan", "han", "haa", "haanji", "hanji",
        "theek", "thik", "chalo", "confirm", "kardo", "karde", "bilkul", "zaroor", "jaroor",
        # "haan ji" as speech recognition writes it: the h is often lost ("अंजी" in the real test)
        "anji", "aanji", "haanjee", "hanjee", "anjee",
        "हाँ", "हां", "ठीक", "बिल्कुल", "ਹਾਂ", "ਠੀਕ", "अंजी", "आंजी", "हांजी", "हाँजी", "ਹਾਂਜੀ"}
_NO = {"no", "n", "nope", "nah", "nahi", "nahin", "nai", "nahiin", "mat", "cancel", "stop", "ruk", "ruko",
       "rehne", "rehnede", "rehnedo", "rahne", "rahnedo", "rehan", "rahan", "rehn", "rahin", "rehin",  # "rehne do"
       "chhod", "chod", "chhodo", "chodo", "chhoddo", "choddo", "chhad", "chhado", "chad", "chado",  # "chhod do"
       "koni",  # Haryanvi "not"
       "dont", "don't", "नहीं", "नही", "मत", "रुको", "रुक", "रहने", "रहन", "छोड़", "छोड़ो", "छोड़", "छोड़ो", "कोनी",
       "ਨਹੀਂ", "ਨਹੀ", "ਨਾ", "ਰਹਿਣ", "ਛੱਡ", "ਛੱਡੋ"}
_YES_PHRASES = ("go ahead", "do it", "kar do", "kar de", "kar dijiye", "kar dio", "kar diyo", "theek hai",
                "thik hai", "ha ji", "haa ji", "कर दो", "कर दे", "ठीक है", "हा जी", "ਕਰ ਦਿਓ")
_NO_PHRASES = ("never mind", "leave it", "forget it", "not now", "no thanks")
_SOFTENED = {"kar", "karo", "de", "do", "dena", "dijiye", "dio", "diyo", "kardo", "karde"}  # "... na" = please


def parse_confirmation(text: str) -> bool | None:
    """True for a clear yes, False for a no, None for anything else (treated as a new request).

    A no always wins: "haan... nahi, rehne do" cancels, and so does a short answer with a no in
    it ("nahi renny do" - a misheard "rehne"). Other words in a longer answer make it a new
    request instead ("yes and also open chrome").
    """
    norm = unicodedata.normalize("NFC", text).lower()
    # Drop punctuation/symbols only; \W would also strip Devanagari/Gurmukhi vowel signs.
    norm = "".join(" " if unicodedata.category(c)[0] in "PS" and c != "'" else c for c in norm)
    words = norm.split()
    if not words or len(words) > 6:
        return None
    joined = " ".join(words)
    yes = any(w in _YES for w in words) or any(p in joined for p in _YES_PHRASES)
    no = any(w in _NO for w in words) or any(p in joined for p in _NO_PHRASES)
    # "na" / "ना" alone (or "na ji") is a no; after a verb it softens a yes ("kar de na").
    if ("na" in words or "ना" in words) and not set(words) & _SOFTENED:
        no = True
    phrase_words = {w for p in _YES_PHRASES + _NO_PHRASES if p in joined for w in p.split()}
    content = [w for w in words if w not in _FILLER and w not in _YES and w not in _NO and w not in phrase_words
               and w != "ना"]
    if no and len(words) <= 4:
        return False  # a short answer with a no in it is a no, even with a word the model misheard
    if content:  # other words present: this is a new request, not an answer
        return None
    if no:
        return False
    return True if yes else None


def _matches(patterns: list[re.Pattern[str]], text: str) -> bool:
    for pattern in patterns:
        for match in pattern.finditer(text):
            if not _NEGATED.search(text[max(0, match.start() - 100):match.start()]):
                return True
    return False


def claims_success(text: str) -> bool:
    """Says something is done/ready/opened. Only valid right after a verified action."""
    return _matches(_COMPLETION, text)


def claims_unverified(text: str) -> bool:
    """Anything a conversation-only reply must not say: done, doing it now, please wait, 'are you sure?'."""
    return _matches(_COMPLETION, text) or _matches(_ACTIVITY, text)


def _unit(raw: str) -> str:
    unit = raw.lower().replace("°", "").replace(" ", "")
    return "c" if unit in ("", "c", "degree", "degrees") else unit


def unverified_figures(text: str, context: str = "") -> list[str]:
    """Hardware figures in `text` whose kind (GB, %, C...) has no verified figure in `context`."""
    verified_units = {_unit(m.group(2)) for m in _FIGURE.finditer(context)}
    return [m.group(0) for m in _FIGURE.finditer(text) if _unit(m.group(2)) not in verified_units]


class ResponseComposer:
    def guard(self, text: str, *, language: str, context: str = "") -> tuple[str, bool]:
        """For conversation-only replies: (safe text, was_rewritten).

        `context` is what NEXUS already reported this conversation (tool results),
        so follow-ups like "is 50 C ok?" can discuss real numbers.
        """
        text = text.strip()
        if not text:
            return text, False
        if claims_unverified(text):
            return _NOTHING_DONE.get(language, _NOTHING_DONE["en"]), True
        if unverified_figures(text, context):
            return _UNCHECKED.get(language, _UNCHECKED["en"]), True
        return text, False

    def acknowledge(self, language: str, user_text: str) -> str:
        """Code-written acknowledgement before actions, mirroring 'bhai'/'yaar'."""
        words = set(re.findall(r"\w+", user_text.lower()))
        friend = "bhai" if "bhai" in words else "yaar" if "yaar" in words else ""
        if language == "hinglish":
            return f"Haan {friend}, ek second." if friend else "Haan, ek second."
        if language == "hi":
            return "ठीक है, एक सेकंड।"
        if language == "pa":
            return "Haan ji, ik second."
        return f"Sure {friend}, one moment." if friend else "Sure, one moment."

    def announce(self, tool: Tool, args: Mapping[str, Any]) -> str | None:
        if tool.progress is None:
            return None
        try:
            text = tool.progress(args).strip()
        except Exception:
            return None
        return (text[:1].upper() + text[1:] + "...") if text else None

    def confirmation_prompt(self, outcome: Outcome, language: str) -> str:
        return f"{outcome.message} {_ASK.get(language, _ASK['en'])}"

    def result(self, outcome: Outcome, language: str) -> Say:
        """SUCCESS / FAILURE / REFUSAL / UNSUPPORTED are worded differently: an intentional
        security refusal is never described as a failed operation."""
        s, message = outcome.status, outcome.message
        if s is Status.DONE:
            return Say(SayKind.RESULT, message, language)
        if s is Status.FAILED:  # attempted, didn't work
            if message.startswith(("I couldn't", "I could not")):
                return Say(SayKind.ERROR, message, language)
            return Say(SayKind.ERROR, f"That didn't work. {message}", language)
        if s is Status.BLOCKED:  # NEXUS won't: the policy or the tool refused on purpose
            if message.startswith("Refused: "):
                message = "I can't do that: " + message[len("Refused: "):]
            return Say(SayKind.REFUSAL, message, language)
        if s is Status.UNSUPPORTED:
            return Say(SayKind.UNSUPPORTED, message, language)
        if s is Status.INVALID:
            return Say(SayKind.ERROR, f"I couldn't do that: {message}", language)
        return Say(SayKind.RESULT, message, language)  # declined / expired
