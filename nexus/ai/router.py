"""Response paths: answer trivial requests without loading any language model.

    "what time is it?"              -> fast path: get_datetime       (no model)
    "GPU ka temperature kya hai?"   -> fast path: get_system_info    (no model)
    "remember that Chrome is my..." -> fast path: remember           (no model)
    "what do you remember about me" -> fast path: list_memories      (no model)
    "open chrome", "yaar I'm bored" -> planner (language model)

The fast path is deliberately high-precision: a request is only taken if it
clearly matches and contains nothing else to do ("and", "open", "search"...).
Anything uncertain goes to the planner, which can do everything the fast path
does - just more slowly and with a model loaded.
"""

from __future__ import annotations

import re
import unicodedata

from nexus.ai.conversation.types import ActionStep, Plan
from nexus.system.apps.names import TRAILING_VERBS

_HINGLISH = {"kya", "hai", "hain", "kitna", "kitni", "kitne", "batao", "bata", "mera", "meri", "mere", "abhi",
             "bhai", "yaar", "kar", "karo", "kaise", "kaisa", "kaunsa", "konsa", "aaj", "ka", "ki", "ke",
             "mein", "yaad", "rakh", "rakhna", "tum", "tumhe", "hua", "baje", "bachi", "bacha", "jagah"}


def detect_language(text: str) -> str:
    if any("ऀ" <= c <= "ॿ" for c in text):
        return "hi"
    if any("਀" <= c <= "੿" for c in text):
        return "pa"
    words = set(re.findall(r"[a-z]+", text.lower()))
    return "hinglish" if len(words & _HINGLISH) >= 1 else "en"


# --- patterns ----------------------------------------------------------------------------

_TIME = re.compile(
    r"\bwhat(?:'s| is)?\s+(?:the\s+)?time\b|\btime\s+(?:is it|now|please)\b|\bcurrent time\b|"
    r"\bwhat(?:'s| is)?\s+(?:the\s+|today'?s\s+)?date\b|\bwhat day is (?:it|today)\b|\btoday'?s date\b|"
    r"\btime\s+kya\s+(?:hua|hai|ho\s*gaya)\b|\bkitne\s+baje\b|\babhi\s+(?:ka\s+)?time\b|"
    r"\bkya\s+time\s+(?:hai|hua|ho\s*(?:raha|rha|gaya|gya))\b|"
    r"\baaj\s+(?:ki\s+|kaun\s*si\s+|kya\s+)?(?:date|tareekh|tarikh)\b|\baaj\s+kaun\s*sa\s+din\b|"
    r"\bkinne\s+vaje\b|कितने\s+बजे|समय\s+क्या|आज\s+(?:की\s+)?तारीख|ਕਿੰਨੇ\s+ਵਜੇ", re.IGNORECASE)

_COMPONENTS = {
    "gpu": ("gpu", "graphics", "vram", "video card", "graphic card"),
    "cpu": ("cpu", "processor", "processer"),
    "memory": ("ram",),
    "storage": ("storage", "disk", "ssd", "hdd", "space", "jagah", "c drive", "d drive"),
}
_WHOLE_SYSTEM = ("system", "pc", "computer", "laptop")
_METRIC = {"temp", "temperature", "garam", "usage", "use", "used", "using", "load", "busy", "kitna", "kitni",
           "kitne", "much", "many", "free", "left", "bachi", "bacha", "khali", "status", "halat", "haal",
           "health", "info", "details", "which", "what", "kaunsa", "konsa", "kaisa", "kaisi", "check", "batao"}
_NOT_A_QUERY = {"open", "close", "kill", "band", "khol", "kholo", "chala", "chalu", "start", "delete", "remove",
                "install", "uninstall", "search", "google", "youtube", "dhundh", "format", "overclock", "set",
                "change", "increase", "decrease", "badha", "ghata", "restart", "shutdown", "update", "download",
                "and", "aur", "phir", "then", "also", "why", "kyun", "kyu", "should", "upgrade", "buy", "fix",
                "remember", "yaad", "clean", "free up", "khali kar"}

# "what is RAM?" / "GPU kya hota hai" ask for knowledge, not a reading of this machine.
_DEFINITION = re.compile(
    r"\bwhat\s+(?:is|are)\s+(?:a\s+|an\s+)?(?:ram|gpu|cpu|vram|ssd|hdd|storage|processor)s?\s*$|"
    r"\b(?:hota|hoti|hote|matlab|meaning|means|mean|explain|difference|vs|versus)\b", re.IGNORECASE)

_LIST_MEMORIES = re.compile(
    r"\bwhat\s+(?:do|did)\s+you\s+(?:remember|know)\s+about\s+me\b|\bwhat\s+have\s+you\s+remembered\b|"
    r"\b(?:show|list)\s+(?:me\s+)?(?:my\s+|your\s+)?memories\b|"
    r"\b(?:tumhe|tujhe|aapko|tuje|tereko|tumko)?\s*mere\s+(?:baare|bare)\s+(?:mein|me|main)\s+kya\s+(?:kya\s+)?"
    r"(?:yaad|pata)\s+hai\b|\bkya\s+kya\s+yaad\s+hai\b", re.IGNORECASE)

_REMEMBER = [
    re.compile(r"^(?:please\s+|nexus\s+)?remember(?:\s+that)?\s+(?P<fact>.{3,})$", re.IGNORECASE),
    re.compile(r"^yaad\s+rakh(?:na|o|iye|ho)?(?:\s+(?:ki|ke|kee))?\s+(?P<fact>.{3,})$", re.IGNORECASE),
    re.compile(r"^(?P<fact>.{3,}?)[\s,.;-]+(?:ye\s+|yeh\s+|isko\s+|ise\s+)?yaad\s+rakh(?:na|o|iye|ho)?$",
               re.IGNORECASE),
]
_TEMPORARY = re.compile(r"\b(?:for today|for now|temporarily|aaj ke liye|abhi ke liye)\b", re.IGNORECASE)
_PRIVATE = re.compile(r"\b(?:privately|as private|private mein|ye private hai)\b", re.IGNORECASE)


def _words(text: str) -> list[str]:
    norm = unicodedata.normalize("NFC", text.lower())
    return "".join(c if unicodedata.category(c)[0] in "LMN" or c in " '" else " " for c in norm).split()


_LAUNCH_LEAD = re.compile(r"^\s*(?:please\s+)?(?:open|start|launch|run|play|fire\s+up|boot\s+up|load\s+up|"
                          r"bring\s+up|switch\s+to)\s+(?P<name>.+?)\s*[.!?]*$", re.IGNORECASE)
_LAUNCH_TRAIL = re.compile(rf"^\s*(?P<name>.+?)\s+{TRAILING_VERBS}"
                           r"(?:\s+(?:na|please|zara|bhai|yaar|jaldi|re|ji))*\s*[.!?]*$", re.IGNORECASE)
_GO_TO = re.compile(r"^\s*(?:go\s+to|visit|take\s+me\s+to)\s+(?P<name>.+?)\s*[.!?]*$", re.IGNORECASE)
_DOMAIN = re.compile(r"^(?:https?://)?(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|org|net|in|io|dev|co|gov|edu|"
                     r"tv|me|ai|app|gg|info|uk)(?:/\S*)?$", re.IGNORECASE)
_COMPOUND = re.compile(r"\b(?:and|aur|then|phir|also)\b|,", re.IGNORECASE)
_NAME_FILLER = re.compile(r"^(?:(?:nexus|bhai|yaar|zara|please|mera|meri|the|my)\s+)+", re.IGNORECASE)
# Websites people name without a domain ("go to youtube"). Only used when no installed app has the name.
KNOWN_SITES = {
    "youtube": "https://www.youtube.com", "google": "https://www.google.com", "gmail": "https://mail.google.com",
    "github": "https://github.com", "wikipedia": "https://www.wikipedia.org", "netflix": "https://www.netflix.com",
    "instagram": "https://www.instagram.com", "facebook": "https://www.facebook.com", "reddit": "https://www.reddit.com",
    "amazon": "https://www.amazon.in", "flipkart": "https://www.flipkart.com", "chatgpt": "https://chatgpt.com",
    "twitter": "https://x.com", "linkedin": "https://www.linkedin.com", "whatsapp web": "https://web.whatsapp.com",
}
# Folder words that are never app names; "photos"/"music"/"videos" may be apps, so those check apps first.
_PLAIN_FOLDERS = {"downloads", "download", "documents", "desktop", "this pc", "my computer", "home"}

_FORCE_CLOSE = re.compile(r"^\s*(?:please\s+)?(?:force[\s-]*(?:close|quit|stop)|kill|end\s+task(?:\s+for)?)\s+"
                          r"(?P<name>.+?)\s*[.!?]*$", re.IGNORECASE)
_CLOSE_LEAD = re.compile(r"^\s*(?:please\s+)?(?:close|quit|exit|shut\s+down|stop)\s+(?P<name>.+?)\s*[.!?]*$",
                         re.IGNORECASE)
_CLOSE_TRAIL = re.compile(r"^\s*(?P<name>.+?)\s+band\s+kar(?:\s*(?:o|do|de|na|dijiye|dena))?"
                          r"(?:\s+(?:na|please|zara|bhai|yaar|jaldi|re|ji))*\s*[.!?]*$", re.IGNORECASE)
_ALIAS_ALIAS = r"(?P<alias>[^\W_][^\W_ ]*(?: [^\W_]+){0,3})"
_ALIAS_SET = [
    re.compile(rf"^\s*(?:remember\s+(?:that\s+)?)?(?P<app>.+?)\s+is\s+my\s+{_ALIAS_ALIAS}\s*[.!]*$", re.IGNORECASE),
    re.compile(rf"^\s*(?:please\s+)?call\s+(?P<app>.+?)\s+my\s+{_ALIAS_ALIAS}\s*[.!]*$", re.IGNORECASE),
    re.compile(rf"^\s*(?P<app>.+?)\s+(?:mera|meri)\s+{_ALIAS_ALIAS}\s+ha(?:i|y)\s*[.!]*$", re.IGNORECASE),
]
_ALIAS_REMOVE = re.compile(rf"^\s*(?:please\s+)?(?:forget|remove|delete)\s+(?:my\s+|the\s+)?{_ALIAS_ALIAS}\s+"
                           r"(?:alias|nickname|app\s+name|shortcut\s+name)\s*[.!]*$", re.IGNORECASE)


class FastPathRouter:
    def __init__(self, apps=None) -> None:
        self.apps = apps  # AppService (or anything with resolve()); None = no launch fast path

    def route(self, text: str) -> Plan | None:
        """A plan that needs no language model, or None to use the planner."""
        language = detect_language(text)
        step = (self._alias(text) or self._close(text) or self._remember(text) or self._memories(text)
                or self._time(text) or self._system(text) or self._open(text))
        if step is None:
            return None
        return Plan(say="", actions=(step,), language=language, acknowledge=False)

    def _known(self, name: str) -> bool:
        if self.apps is None:
            return False
        res = self.apps.resolve(name)
        return res.confident or res.ambiguous

    def _alias(self, text: str) -> ActionStep | None:
        if self.apps is None:
            return None
        removal = _ALIAS_REMOVE.match(text)
        if removal:
            return ActionStep("remove_app_alias", {"alias": removal.group("alias")})
        for pattern in _ALIAS_SET:
            match = pattern.match(text)
            if match:
                app = _NAME_FILLER.sub("", match.group("app")).strip(" \"'")
                # Only when the named thing really is an installed app; otherwise it's a memory/fact.
                res = self.apps.catalog().resolve(app)
                if res.confident or res.ambiguous:
                    return ActionStep("set_app_alias", {"app": app, "alias": match.group("alias")})
        return None

    def _close(self, text: str) -> ActionStep | None:
        from nexus.tools.apps import _EXECUTABLE_LIKE, _PATH_LIKE, _PID_LIKE

        forced = _FORCE_CLOSE.match(text)
        match = forced or _CLOSE_LEAD.match(text) or _CLOSE_TRAIL.match(text)
        if match is None:
            return None
        name = _NAME_FILLER.sub("", match.group("name")).strip(" \"'")
        if not name or _COMPOUND.search(name) or len(name.split()) > 6:
            return None
        tool = "force_close_app" if forced else "close_app"
        if _PID_LIKE.match(name) or _EXECUTABLE_LIKE.search(name) or _PATH_LIKE.search(name):
            return ActionStep(tool, {"name": name})  # the tool refuses it - no model needed for that
        return ActionStep(tool, {"name": name}) if self._known(name) else None

    def _open(self, text: str) -> ActionStep | None:
        from nexus.tools.apps import _EXECUTABLE_LIKE, _PATH_LIKE, folder_role

        go_to = _GO_TO.match(text)
        match = go_to or _LAUNCH_LEAD.match(text) or _LAUNCH_TRAIL.match(text)
        if match is None:
            return None
        name = _NAME_FILLER.sub("", match.group("name")).strip(" \"'")
        if not name or _COMPOUND.search(name) or len(name.split()) > 6:
            return None
        low = name.lower()
        if _DOMAIN.match(low):
            return ActionStep("open_url", {"url": low if low.startswith("http") else f"https://{low}"})
        if go_to:
            site = KNOWN_SITES.get(low)
            return ActionStep("open_url", {"url": site}) if site else None
        if _EXECUTABLE_LIKE.search(name) or _PATH_LIKE.search(name):
            return ActionStep("launch_app", {"name": name})  # the tool refuses it - no model needed for that
        role = folder_role(name)
        explicit_folder = re.search(r"\bfolder\b", low) is not None
        if role and (explicit_folder or " ".join(_words(name)) in _PLAIN_FOLDERS):
            return ActionStep("open_folder", {"target": name})
        if self.apps is not None:
            res = self.apps.resolve(name)
            if res.confident or res.ambiguous:  # ambiguous: the tool asks which one - still no model
                return ActionStep("launch_app", {"name": name})
        if role:
            return ActionStep("open_folder", {"target": name})
        site = KNOWN_SITES.get(low)
        return ActionStep("open_url", {"url": site}) if site else None

    def _time(self, text: str) -> ActionStep | None:
        if _TIME.search(text) and not self._has_other_request(text, allow={"what", "is", "it"}):
            return ActionStep("get_datetime", {})
        return None

    def _system(self, text: str) -> ActionStep | None:
        low = " " + " ".join(_words(text)) + " "
        words = set(low.split())
        if len(words) > 12 or words & _NOT_A_QUERY or any(f" {p} " in low for p in _NOT_A_QUERY if " " in p):
            return None
        if _DEFINITION.search(text.strip().rstrip("?!. ")):
            return None
        sections = {s for s, keys in _COMPONENTS.items() if any(f" {k} " in low for k in keys)}
        whole = any(f" {k} " in low for k in _WHOLE_SYSTEM)
        if not (words & _METRIC) or not (sections or whole):
            return None
        section = sections.pop() if len(sections) == 1 else "all"
        return ActionStep("get_system_info", {"section": section})

    def _memories(self, text: str) -> ActionStep | None:
        return ActionStep("list_memories", {}) if _LIST_MEMORIES.search(text) else None

    def _remember(self, text: str) -> ActionStep | None:
        body = text.strip().rstrip(".!?")
        for pattern in _REMEMBER:
            match = pattern.match(body)
            if match is None:
                continue
            fact = match.group("fact")
            if re.match(r"(?:to|ki mujhe|mujhe)\s", fact, re.IGNORECASE):
                return None  # "remember to buy milk" is a reminder, not a fact - let the planner answer
            scope = "device"
            if _TEMPORARY.search(fact):
                scope, fact = "temporary", _TEMPORARY.sub("", fact)
            if _PRIVATE.search(fact):
                scope, fact = "private", _PRIVATE.sub("", fact)
            # "remember privately that X": the scope word came first, so "that" is still leading
            fact = re.sub(r"^(?:that|ki|ke)\s+", "", fact.strip(" ,.;:-"), flags=re.IGNORECASE)
            if len(fact) >= 3:
                return ActionStep("remember", {"text": fact, "scope": scope})
        return None

    @staticmethod
    def _has_other_request(text: str, allow: set[str]) -> bool:
        words = set(_words(text))
        return bool(words & (_NOT_A_QUERY - allow))
