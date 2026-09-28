"""NEXUS text console.

Plain text is a conversation with NEXUS (through the conversation engine and
Action Kernel). Lines starting with "/" are developer commands, including
/run to call a tool directly.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable

from nexus import __version__
from nexus.ai.conversation.types import ActionReport, ConfirmationRequest, Say, SayKind, TurnEvent, UserInput
from nexus.ai.providers.base import ChatMessage
from nexus.ai.voice import SilentTTS, TTSProvider
from nexus.ai.voice.speaker import Speaker
from nexus.core.kernel import Outcome, Status, ToolRequest
from nexus.core.runtime import Runtime
from nexus.tools.base import Risk

HELP = r"""Talk to NEXUS in English, Hindi, Hinglish or Punjabi - or use a command:
  /tools                         list tools and their risk level
  /help <tool>                   show a tool's arguments
  /run <tool> [name=value ...]   run a tool directly through the security policy
  /status                        NEXUS state, memory and VRAM use, loaded AI models
  /gamemode on|off               free the GPU for a game (AI falls back to CPU)
  /unload                        unload all AI models now
  /apps [games|refresh]          installed apps/games NEXUS found (refresh = scan again)
  /aliases, /unalias <name>      your own app names ("Chrome is my browser") and removing one
  /listen  (or Enter on an empty line)   say something - NEXUS stops at your pause; Ctrl+C cancels
  /mic [use <name> | auto]       microphones NEXUS can use, and which one it picked
  /tts [on | off | test]         voice output: status, turn it on/off, say a test sentence
  /stop                          stop speaking now (typing or talking also interrupts NEXUS)
  /memory                        everything NEXUS remembers (with ids)
  /forget <id>                   delete one memory by id (first characters are enough)
  /audit [n]  |  /audit verify   show / check the audit log
  /quit                          exit

Quote /run values that contain spaces:  name="some value"
  /run get_system_info section=gpu
  /run create_folder path=D:\nexus-test        (outside your safe folders: asks first)"""

_ARG = re.compile(r"""\s*([A-Za-z_]\w*)=(?:"([^"]*)"|'([^']*)'|(\S+))""")

_LABELS = {
    Status.DONE: "done",
    Status.FAILED: "FAILED",
    Status.DECLINED: "cancelled",
    Status.BLOCKED: "REFUSED",
    Status.INVALID: "invalid",
    Status.EXPIRED: "expired",
    Status.NEEDS_CONFIRMATION: "needs your OK",
    Status.NEEDS_CLARIFICATION: "needs your choice",
    Status.UNSUPPORTED: "not supported yet",
}


def parse_args(text: str) -> dict[str, str]:
    args: dict[str, str] = {}
    text = text.strip()
    pos = 0
    while pos < len(text):
        match = _ARG.match(text, pos)
        if match is None:
            raise ValueError(f"couldn't read {text[pos:].strip()!r} - use name=value, "
                             "and put quotes around values with spaces")
        name, *values = match.groups()
        args[name] = next(v for v in values if v is not None)
        pos = match.end()
    return args


class Console:
    def __init__(self, runtime: Runtime, assistant=None, *, input_fn: Callable[[str], str] = input,
                 output: Callable[[str], None] = print, tts: TTSProvider | None = None) -> None:
        self.rt = runtime
        self.assistant = assistant
        self.speaker = getattr(assistant, "speaker", None) or Speaker(tts or SilentTTS())
        self._input = input_fn
        self._out = output

    def run(self) -> None:
        self._out(f"NEXUS {__version__}")
        if self.assistant is None:
            self._out("AI layer is off - only /commands work.")
        self._out("Type to talk to NEXUS, or /help for commands.\n")
        commands = {"help": self._help, "tools": self._tools, "run": self._run, "audit": self._audit,
                    "status": self._status, "gamemode": self._gamemode, "unload": self._unload,
                    "memory": self._memory, "forget": self._forget, "apps": self._apps,
                    "aliases": self._aliases, "unalias": self._unalias, "listen": self._listen,
                    "l": self._listen, "mic": self._mic, "tts": self._tts, "stop": self._stop_speaking}
        apps = getattr(self.rt, "apps", None)
        if apps is not None:
            apps.subscribe(self._on_app_event)  # e.g. "GTA V is running." after a background launch
        if self._voice is not None:
            # "Microphone connected: ..." as soon as it is plugged in - no restart needed.
            self._voice.mics.subscribe(lambda change: self._out(f"  ({change.describe()})"))
            self._voice.mics.scan()
            self._voice.mics.start_watching(self._voice.device_poll_s)
        while True:
            try:
                # Windows PowerShell prefixes piped text with a byte-order mark.
                line = self._input("you> ").replace("\ufeff", "").strip()
            except (EOFError, KeyboardInterrupt):
                self._out("")
                return
            self._interrupt_speech()  # anything new - typing, a command, push-to-talk - interrupts NEXUS
            if not line:
                if self._voice is not None:
                    self._listen("")  # push-to-talk: Enter on an empty line
                continue
            if not line.startswith("/"):
                self._chat(line)
                continue
            cmd, _, rest = line[1:].partition(" ")
            cmd = cmd.lower()
            if cmd in ("quit", "exit"):
                return
            handler = commands.get(cmd)
            if handler is None:
                self._out(f"  Unknown command '/{cmd}'. Type /help.")
            else:
                handler(rest.strip())

    # --- conversation ------------------------------------------------------------

    @property
    def _voice(self):
        return getattr(self.assistant, "voice", None)

    def _chat(self, text: str, *, heard=None) -> None:
        if self.assistant is None:
            self._out("  The AI layer isn't running. Use /run to call tools directly.")
            return
        user = (UserInput(text, interface="console") if heard is None
                else UserInput(heard.text, interface="voice", confidence=heard.confidence,
                               inferred=getattr(heard, "guessed", False)))
        prepare = getattr(self.speaker, "prepare", None)
        if prepare is not None:
            prepare()  # load the voice while NEXUS thinks, so the answer is heard sooner
        self.assistant.engine.handle(user, self._show_event)
        reply = getattr(self.assistant.engine.planner, "last_reply", None)
        if reply is not None and self.assistant.chat.last_choice:
            self._out(f"  ({self.assistant.chat.last_choice}, {reply.seconds:.1f}s)")
            self.assistant.engine.planner.last_reply = None

    def _show_event(self, event: TurnEvent) -> None:
        if isinstance(event, Say):
            lines = event.text.splitlines() or [""]
            self._out(f"nexus> {lines[0]}")
            for line in lines[1:]:
                self._out(f"       {line}")
            try:
                self.speaker(event)  # an output layer: it can never break the conversation
            except Exception:
                logging.getLogger(__name__).exception("voice output failed")
        elif isinstance(event, ActionReport):
            unverified = event.status is Status.DONE and event.verified is None and event.tool != "launch_app"
            note = "  (not independently verifiable)" if unverified else ""
            self._out(f"  [{_LABELS.get(event.status, event.status.value)}] {event.tool}{note}")
        elif isinstance(event, ConfirmationRequest):
            self._out("  (answer yes/haan or no/nahi)")

    def _on_app_event(self, event) -> None:
        """A background result arrived (from another thread): show it like any other reply."""
        say = Say(SayKind.RESULT if event.ok else SayKind.ERROR, event.message)
        self._show_event(say)
        if self.assistant is not None:
            self.assistant.engine.history.append("default", ChatMessage("assistant", event.message))

    # --- voice output ---------------------------------------------------------------

    def _interrupt_speech(self) -> None:
        try:
            self._interrupted = bool(self.speaker.stop())
        except Exception:
            logging.getLogger(__name__).exception("stopping speech failed")

    def _stop_speaking(self, _: str) -> None:
        # The new input line has already interrupted it (see run()); this only reports.
        self._out("  (NEXUS stopped speaking)" if getattr(self, "_interrupted", False) else "  (NEXUS wasn't speaking)")

    def _tts(self, rest: str) -> None:
        speaker = self.speaker
        cmd, _, arg = rest.partition(" ")
        if cmd in ("on", "off"):
            speaker.enabled = cmd == "on"
            self._out(f"  Voice output {'on' if speaker.enabled else 'off (text only)'}.")
            return
        if cmd == "test":
            if not hasattr(speaker, "say") or not speaker.enabled:
                self._out("  Voice output is off.")
                return
            text = arg.strip() or "Hello, I'm Nexus. This is how I sound."
            if speaker.say(text):
                last = speaker.last
                self._out(f"  (spoke {last.get('characters')} characters; first sound after "
                          f"{last.get('first_audio_s')} s, done in {last.get('total_s')} s"
                          + (f", including {last['load_s']} s to load the voice" if last.get("load_s") else "") + ")")
            else:
                why = getattr(speaker, "problem", "") or getattr(self.assistant, "speech_note", "") or "not available"
                self._out(f"  Nothing was spoken: {why}")
            return
        if cmd:
            self._out("  usage: /tts   |   /tts on | off   |   /tts test [text]")
            return
        status = speaker.status() if hasattr(speaker, "status") else {}
        note = getattr(self.assistant, "speech_note", "")
        self._out(f"  voice output: {'on' if status.get('enabled', True) else 'off'}   provider: "
                  f"{status.get('provider', speaker.tts.info.name)}"
                  + (f" ({'loaded' if status.get('loaded') else 'loads when needed'})" if status else ""))
        if note:
            self._out(f"  text only: {note}")
        if status.get("output"):
            self._out(f"  output: {status['output']} (the Windows default output)")
        elif status.get("output_problem"):
            self._out(f"  output: none - {status['output_problem']}")
        if status.get("problem"):
            self._out(f"  last problem: {status['problem']}")
        last = status.get("last") or {}
        if last.get("total_s") is not None:
            self._out(f"  last reply: {last.get('characters')} characters, first sound after "
                      f"{last.get('first_audio_s')} s, {last.get('total_s')} s in total"
                      + (" (stopped)" if last.get("stopped") else ""))

    # --- voice ---------------------------------------------------------------------

    def _listen(self, _: str) -> None:
        """One spoken request, through exactly the same engine as typed text."""
        from nexus.ai.voice.base import ListeningCancelled, VoiceInputError
        from nexus.system.audio.errors import AudioInputError

        voice = self._voice
        if voice is None:
            note = getattr(self.assistant, "voice_note", "") or "the AI layer is off"
            self._out(f"  Voice input isn't available: {note}.")
            return

        def status(message: str) -> None:
            if message.startswith("listening with"):
                self._out(f"  ({message} - speak now; Ctrl+C cancels)")
            elif message != "transcribing":
                self._out(f"  ({message})")

        try:
            heard = voice.listen(on_status=status)
        except ListeningCancelled:
            self._out("  (cancelled)")
            return
        except KeyboardInterrupt:
            self._out("  (cancelled)")
            return
        except (VoiceInputError, AudioInputError) as exc:
            self._out(f"  (voice) {exc}")
            return
        except Exception as exc:  # a driver or audio-library failure must never end the console
            logging.getLogger(__name__).exception("voice input failed")
            self._out(f"  (voice) Voice input failed unexpectedly ({type(exc).__name__}: {exc}). "
                      "Typing still works; details are in var/logs/nexus.log.")
            return
        t = voice.last_timings
        language = heard.transcript.language or "?"
        romanized = f' -> "{heard.text}"' if heard.text != heard.original else ""
        confidence = f"{heard.confidence:.2f}" if heard.confidence is not None else "?"
        self._out(f'  heard: "{heard.original}"{romanized}   ({language}, confidence {confidence}, '
                  f"transcribed in {t.get('transcribing_s', 0):.1f}s)")
        guesses = [c for c in getattr(heard, "changes", ()) if "sounds alike" in c]
        if guesses:
            self._out(f"  (understood by sound: {'; '.join(guesses)} - NEXUS will ask before acting)")
        if heard.truncated:
            self._out("  (you spoke for the maximum length, so the end may be cut off)")
        self._chat(heard.text, heard=heard)

    def _mic(self, rest: str) -> None:
        voice = self._voice
        if voice is None:
            note = getattr(self.assistant, "voice_note", "") or "the AI layer is off"
            self._out(f"  Voice input isn't available: {note}.")
            return
        cmd, _, arg = rest.partition(" ")
        if cmd == "use" and arg.strip():
            ok, message = voice.mics.use(arg.strip())
            self._out(f"  {message}")
            return
        if cmd == "auto":
            self._out(f"  {voice.mics.use('auto')[1]}")
            return
        if cmd not in ("", "rescan"):
            self._out("  usage: /mic   |   /mic use <part of a name>   |   /mic auto")
            return
        for line in voice.mics.describe():
            self._out(line)

    # --- commands ----------------------------------------------------------------

    def _aliases(self, _: str) -> None:
        memory = getattr(self.rt, "memory", None)
        names = memory.aliases() if memory is not None else []
        if not names:
            self._out("  You haven't named any apps yet (e.g. \"Chrome is my browser\").")
        for alias, app in names:
            self._out(f"  your {alias:<20} -> {app}")

    def _unalias(self, rest: str) -> None:
        apps = getattr(self.rt, "apps", None)
        if apps is None or not rest:
            self._out("  usage: /unalias <name>   (see /aliases)")
            return
        out = apps.remove_alias(rest)
        self.rt.audit.record("alias_removed", alias=rest, ok=out.ok, via="console")
        self._out(f"  {out.message}")

    def _help(self, rest: str) -> None:
        if not rest:
            self._out(HELP)
            return
        tool = self.rt.registry.get(rest)
        if tool is None:
            self._out(f"  No tool called '{rest}'. Type /tools to list them.")
            return
        self._out(f"  {tool.name} [{tool.risk.name}] - {tool.description}")
        for p in tool.params:
            extra = [] if p.required else [f"optional, default {p.default}"]
            if p.choices:
                extra.append("one of: " + ", ".join(p.choices))
            self._out(f"    {p.name} ({p.kind}): {p.description}"
                      + (f" [{'; '.join(extra)}]" if extra else ""))

    def _tools(self, _: str) -> None:
        for tool in self.rt.registry:
            note = "" if tool.risk is Risk.SAFE else "  (asks first)"
            self._out(f"  {tool.name:<18} {tool.risk.name:<8} {tool.description}{note}")

    def _run(self, rest: str) -> None:
        name, _, argtext = rest.partition(" ")
        if not name:
            self._out("  usage: /run <tool> [name=value ...]")
            return
        try:
            args = parse_args(argtext)
        except ValueError as exc:
            self._out(f"  {exc}")
            return
        outcome = self.rt.kernel.submit(ToolRequest(name, args, source="console"))
        while outcome.status is Status.NEEDS_CONFIRMATION:
            self._out(f"  ? {outcome.message}")
            try:
                answer = self._input("  Continue? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            outcome = self.rt.kernel.resolve(outcome.confirmation_id or "", answer in ("y", "yes"))
        self._show_outcome(outcome)

    def _show_outcome(self, outcome: Outcome) -> None:
        self._out(f"  [{_LABELS.get(outcome.status, outcome.status.value)}]")
        for line in outcome.message.splitlines():
            self._out(f"  {line}")
        if outcome.result is not None and outcome.result.ok and outcome.result.verified is None:
            self._out("  (NEXUS can't check this result itself.)")

    def _status(self, _: str) -> None:
        if self.assistant is None:
            self._out("  AI layer is off.")
            return
        a = self.assistant
        st = a.resources.status()
        r = st["resources"]
        self._out(f"  state: {st['state']}   device: {a.device.identity.name} "
                  f"({a.capabilities.role.value}, {a.device.device_id})")
        battery = ("no battery" if r["on_battery"] is None
                   else f"battery {r['battery_percent']:.0f}%{' (on battery)' if r['on_battery'] else ''}")
        self._out(f"  CPU {r['cpu_percent']:.0f}%   GPU {r['gpu_percent'] or 0:.0f}%   {battery}   "
                  f"network {'up' if r['network_up'] else 'offline'}")
        vram = (f"{r['vram_used_mb'] / 1024:.1f} of {r['vram_total_mb'] / 1024:.1f} GB used"
                if r["vram_total_mb"] else "no GPU")
        self._out(f"  RAM {r['ram_available_mb'] / 1024:.1f} of {r['ram_total_mb'] / 1024:.1f} GB free   "
                  f"VRAM {vram}   NEXUS itself {r['nexus_rss_mb']} MB")
        for m in st["models"]:
            state = "loaded" if m["loaded"] else "not loaded"
            size = f"{m['vram_mb']} MB VRAM" if m["placement"] == "gpu" else f"{m['ram_mb']} MB RAM"
            tail = f", unloads in {m['unloads_in_s']}s" if m["unloads_in_s"] is not None else ""
            if m["kept_loaded"]:
                tail = ", kept loaded (your setting)"
            if m["name"] not in a.capabilities.supported_models:
                tail += ", too big for this device"
            self._out(f"    {m['name']:<24} {state:<11} ~{size}{tail}")
        voice = getattr(a, "voice", None)
        if voice is None:
            self._out(f"  voice input: off ({getattr(a, 'voice_note', '') or 'not configured'})")
        else:
            scan = voice.mics.last_scan or voice.mics.scan()
            mic = scan.chosen.name if scan.chosen else "no microphone connected"
            self._out(f"  voice input: {mic}   (/listen or Enter to talk, /mic for details)")
        speaker = getattr(a, "speaker", None)
        if getattr(a, "speech_note", ""):
            self._out(f"  voice output: off ({a.speech_note})")
        elif speaker is not None and hasattr(speaker, "status"):
            st = speaker.status()
            where = st.get("output") or f"no output ({st.get('output_problem')})"
            self._out(f"  voice output: {'on' if st['enabled'] else 'off'} - {st['provider']} -> {where}   (/tts)")

    def _gamemode(self, rest: str) -> None:
        if self.assistant is None:
            self._out("  AI layer is off.")
            return
        if rest == "on":
            unloaded = self.assistant.resources.enter_game_mode()
            self._out(f"  Game mode on. Unloaded: {', '.join(unloaded) or 'nothing was loaded'}.")
            self._out("  NEXUS keeps the GPU free and uses CPU-only models until /gamemode off.")
        elif rest == "off":
            self.assistant.resources.exit_game_mode()
            self._out("  Game mode off.")
        else:
            self._out(f"  Game mode is {'on' if self.assistant.resources.game_mode else 'off'}. "
                      "Use /gamemode on|off.")

    def _unload(self, _: str) -> None:
        if self.assistant is None:
            self._out("  AI layer is off.")
            return
        unloaded = self.assistant.resources.unload_all()
        self._out(f"  Unloaded: {', '.join(unloaded) or 'nothing was loaded'}.")

    def _apps(self, rest: str) -> None:
        apps = getattr(self.rt, "apps", None)
        if apps is None:
            self._out("  App discovery is not available.")
            return
        catalog = apps.refresh() if rest == "refresh" else apps.catalog()
        records = catalog.games() if rest == "games" else catalog.records
        for r in records:
            via = f" via {r.launcher}" if r.launcher else ""
            self._out(f"  [{r.kind.value:<5}] {r.display_name}{via}")
        took = f" (scan took {apps.discovery_seconds:.1f}s)" if apps.discovery_seconds else ""
        self._out(f"  {len(records)} entries{took}. Shells and admin tools are listed but NEXUS won't open them.")

    def _memory(self, _: str) -> None:
        if self.rt.memory is None:
            self._out("  Memory is not available.")
            return
        records = self.rt.memory.all()
        if not records:
            self._out("  NEXUS doesn't remember anything yet.")
        for r in records:
            self._out(f"  {r.id[:8]}  [{r.scope.value:<9}] {r.text}")

    def _forget(self, rest: str) -> None:
        """Keyboard control over memory: no AI involved, the user picks the exact record."""
        if self.rt.memory is None or not rest:
            self._out("  usage: /forget <id>   (see /memory)")
            return
        matches = [r for r in self.rt.memory.all() if r.id.startswith(rest.strip())]
        if len(matches) != 1:
            self._out(f"  {'No' if not matches else 'More than one'} memory matches '{rest}'.")
            return
        self.rt.memory.forget(matches[0].id)
        self.rt.audit.record("memory_forgotten", memory_id=matches[0].id, via="console")
        self._out(f"  Forgotten: {matches[0].text}")

    def _audit(self, rest: str) -> None:
        if rest == "verify":
            ok, detail = self.rt.audit.verify()
            self._out(f"  {'OK' if ok else 'TAMPERED'}: {detail}")
            return
        try:
            n = int(rest) if rest else 10
        except ValueError:
            self._out("  usage: /audit [n] | /audit verify")
            return
        for entry in self.rt.audit.tail(n):
            detail = {k: v for k, v in entry.items() if k not in ("ts", "event", "prev", "hash")}
            self._out(f"  {entry['ts'][11:19]}  {entry['event']:<22} "
                      f"{json.dumps(detail, ensure_ascii=False)}")
