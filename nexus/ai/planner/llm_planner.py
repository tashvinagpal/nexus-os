"""Planner: turns what the user said into a Plan (a reply + proposed actions).

The language model is asked for JSON constrained by a schema whose tool
names are exactly the registered tools. Its output is only a *proposal*: the
Action Kernel validates and judges every action, and results are reported by
code, not by the model.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Protocol, Sequence

from nexus.ai.conversation.types import ActionStep, Plan
from nexus.ai.memory import MemoryRecord
from nexus.ai.providers.base import ChatMessage, ChatReply
from nexus.tools.base import Tool

LANGUAGES = ("en", "hi", "hinglish", "pa", "other")
MAX_SAY_CHARS = 600

SYSTEM_PROMPT = """\
You are NEXUS, the AI layer of the user's own computer (the NEXUS operating system). \
You are a friendly, concise companion: usually one or two short sentences.
Reply in the user's language and style: English, Hindi, Hinglish (Hindi in English letters) \
or Punjabi. Mirror their tone - if they say "bhai" or "yaar", you can too.
Spoken messages may have odd spellings, and Haryanvi or Punjabi words - read them as Hinglish: \
"ke" = "kya", "sai"/"se" at the end = "hai", "koni" = "nahi", "mhara" = "mera", "dasso" = "batao", \
"kar dio" = "kar do", "hoya" = "hua".

You can act ONLY through these tools:
{tools}
{unavailable}
Respond with JSON: {{"language": ..., "actions": [{{"tool": ..., "args": {{...}}}}], "say": ...}}
- "language": en, hi (Devanagari), hinglish (Hindi/Hinglish in English letters), pa (Punjabi) or other.
- Use a tool only if it does exactly what the user asked. Then leave "say" empty: NEXUS \
acknowledges and reports the verified result itself.
- If no tool does exactly that, use NO actions - never substitute a different tool (do not check \
system info or open a website instead) - and say honestly, briefly, that you can't do that yet.
- For small talk, questions, feelings or "what can you do?", reply in "say" with no actions.
- In "say", never claim you did, are doing or will now do something, and never ask "are you sure?" \
- the system handles actions and confirmations.
- Paths must be full Windows paths; "~" means the user's home folder, e.g. "~\\\\Desktop\\\\mods".
- Installed programs and games (Chrome, Steam, GTA...) -> launch_app, never open_url. Websites the user \
names as sites -> open_url. Folders -> open_folder.
- Never invent tools. Never put shell commands, file paths or .exe names into launch_app.

Examples:
"bhai GPU kitna garam hai?" -> {{"language": "hinglish", "actions": [{{"tool": "get_system_info", \
"args": {{"section": "gpu"}}}}], "say": ""}}
"kaisa hai bhai?" -> {{"language": "hinglish", "actions": [], "say": "Ekdum badhiya bhai! Tu bata?"}}
"turn off the firewall" -> {{"language": "en", "actions": [], "say": "I can't change security settings."}}
"uninstall discord" -> {{"language": "en", "actions": [], "say": "I can't uninstall apps yet."}}{memories}"""


# Common requests whose tools arrive in later subsystems. While a tool is missing, the model is
# told plainly, so it says "can't yet" instead of substituting another tool (seen live:
# "GTA chala de" -> opening a GTA web page).
NOT_YET = {
    "launch_app": "opening apps and games",
    "close_app": "closing apps and games",
    "delete_path": "deleting files or folders",
    "set_volume": "changing the volume",
    "remember": "remembering things about the user",
}


class PlanError(ValueError):
    """The model's output could not be turned into a plan."""


class Planner(Protocol):
    def plan(self, text: str, *, history: Sequence[ChatMessage] = (),
             memories: Sequence[MemoryRecord] = ()) -> Plan: ...


class _Chat(Protocol):
    def chat(self, messages: Sequence[ChatMessage], *, schema: dict[str, Any] | None = None,
             temperature: float = 0.2) -> ChatReply: ...


def describe_tools(tools: Sequence[Tool]) -> str:
    lines = []
    for tool in tools:
        params = []
        for p in tool.params:
            kind = "|".join(p.choices) if p.choices else p.kind
            params.append(f"{p.name}{'' if p.required else '?'}: {kind}")
        lines.append(f"- {tool.name}({', '.join(params)}): {tool.description}")
    return "\n".join(lines)


def plan_schema(tool_names: Sequence[str], max_actions: int) -> dict[str, Any]:
    # Property order matters for constrained decoding: decide actions, then word the reply.
    return {
        "type": "object",
        "properties": {
            "language": {"type": "string", "enum": list(LANGUAGES)},
            "actions": {
                "type": "array",
                "maxItems": max_actions,
                "items": {
                    "type": "object",
                    "properties": {"tool": {"type": "string", "enum": list(tool_names)},
                                   "args": {"type": "object"}},
                    "required": ["tool", "args"],
                },
            },
            "say": {"type": "string"},
        },
        "required": ["language", "actions", "say"],
    }


def parse_plan(text: str, tools: Sequence[Tool], *, max_actions: int = 3) -> Plan:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlanError(f"model output is not JSON: {exc}") from None
    if not isinstance(data, dict):
        raise PlanError("model output is not a JSON object")

    say = data.get("say")
    say = say.strip()[:MAX_SAY_CHARS] if isinstance(say, str) else ""
    language = data.get("language") if data.get("language") in LANGUAGES else "en"
    raw_actions = data.get("actions") or []
    if not isinstance(raw_actions, list):
        raise PlanError("'actions' is not a list")

    by_name = {t.name: t for t in tools}
    actions = []
    for raw in raw_actions[:max_actions]:
        if not isinstance(raw, dict) or not isinstance(raw.get("tool"), str):
            continue
        args = raw.get("args") if isinstance(raw.get("args"), dict) else {}
        tool = by_name.get(raw["tool"])
        if tool is not None:
            # Drop keys the tool doesn't have and empty optionals the model filled in.
            # This only removes input; the kernel still validates what remains.
            known = {p.name for p in tool.params}
            args = {k: v for k, v in args.items() if k in known and v not in (None, "")}
        actions.append(ActionStep(raw["tool"], args))
    if not say and not actions:
        raise PlanError("model returned an empty plan")
    return Plan(say=say, actions=tuple(actions), language=language)


class LLMPlanner:
    def __init__(self, chat: _Chat, tools: Sequence[Tool], *, max_actions: int = 3,
                 context: Callable[[], str] | None = None) -> None:
        self.chat = chat
        self.tools = list(tools)
        self.max_actions = max_actions
        self.context = context  # verified facts for the prompt, e.g. "Installed games: GTA V, ..."
        self.schema = plan_schema([t.name for t in self.tools], max_actions)
        self._tool_text = describe_tools(self.tools)
        self.last_reply: ChatReply | None = None

    def system_prompt(self, memories: Sequence[MemoryRecord] = ()) -> str:
        mem = ""
        if memories:
            mem = "\n\nThings the user asked you to remember:\n" + "\n".join(f"- {m.text}" for m in memories)
        if self.context is not None:
            extra = self.context()
            if extra:
                mem += "\n\n" + extra
        names = {t.name for t in self.tools}
        missing = [what for tool, what in NOT_YET.items() if tool not in names]
        unavailable = (f"Not available yet: {', '.join(missing)}. For these, say you can't do it yet - "
                       "do not open a website or use another tool instead.\n") if missing else ""
        return SYSTEM_PROMPT.format(tools=self._tool_text, unavailable=unavailable, memories=mem)

    def plan(self, text: str, *, history: Sequence[ChatMessage] = (),
             memories: Sequence[MemoryRecord] = ()) -> Plan:
        messages = [ChatMessage("system", self.system_prompt(memories)), *history, ChatMessage("user", text)]
        self.last_reply = self.chat.chat(messages, schema=self.schema, temperature=0.2)
        return parse_plan(self.last_reply.text, self.tools, max_actions=self.max_actions)
