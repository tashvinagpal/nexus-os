"""Conversation engine: one user turn from words to a verified, spoken answer.

    input -> (pending yes/no?) -> plan (THINKING) -> reply (claim-checked)
          -> for each action: preview -> announce -> kernel (EXECUTING) -> report -> result

Front ends (console, voice, HUD, mesh gateway) pass an `emit` callback and
decide how to show or speak each event.
"""

from __future__ import annotations

import logging
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Sequence

from nexus.ai.conversation.history import HistoryStore, InMemoryHistory
from nexus.ai.conversation.responder import ResponseComposer, parse_confirmation
from nexus.ai.conversation.types import (ActionReport, ActionStep, ConfirmationRequest, Say, SayKind,
                                         TurnEvent, UserInput)
from nexus.ai.memory import MemoryStore
from nexus.ai.planner import PlanError, Planner
from nexus.ai.router import FastPathRouter
from nexus.ai.providers.base import ChatMessage, ProviderUnavailable
from nexus.core.kernel import ActionKernel, Outcome, Status, ToolRequest
from nexus.core.resources import ResourceManager, ResourceState
from nexus.security.permissions import Verdict

log = logging.getLogger(__name__)

Emit = Callable[[TurnEvent], None]

_WAKE = re.compile(r"^\s*(?:(?:hey|hi|ok|okay|oye|oi)\s+)?(?:nexus|नेक्सस|ਨੈਕਸਸ)\b[\s,.:;!-]*", re.IGNORECASE)


def strip_wake_word(text: str) -> str:
    return _WAKE.sub("", text, count=1).strip()


@dataclass(frozen=True)
class _PendingConfirmation:
    confirmation_id: str
    step: ActionStep
    language: str


@dataclass(frozen=True)
class _PendingChoice:
    step: ActionStep
    param: str
    options: tuple[str, ...]
    language: str


# Actions on a named app that the model may only take if the user actually named that app.
_GROUNDED_TOOLS = {"launch_app": "open", "close_app": "close", "force_close_app": "force-close"}

_ORDINALS = {"first": 0, "1": 0, "1st": 0, "one": 0, "pehla": 0, "pehli": 0, "pehle": 0, "pahla": 0,
             "second": 1, "2": 1, "2nd": 1, "two": 1, "dusra": 1, "doosra": 1, "dusri": 1, "dusre": 1,
             "third": 2, "3": 2, "3rd": 2, "three": 2, "teesra": 2, "tisra": 2, "teesri": 2,
             "last": -1, "aakhri": -1, "akhri": -1, "last wala": -1}


def pick_option(text: str, options: Sequence[str]) -> str | None:
    """Which offered option did the user pick? By ordinal ("the second one", "pehla") or by name."""
    from nexus.system.apps.names import aliases_for, normalize, query_variants, score

    words = normalize(text).split()
    # Names and numbers first ("GTA V", "the 4 one", "legacy wala"), then ordinals ("first", "dusra").
    naming = [w for w in words if w not in {"one", "wala", "wali", "vala", "waala", "version"} and w not in _ORDINALS]
    if len(options) == 1 and parse_confirmation(text) is True:  # "Should I open Steam?" -> "haan"
        return options[0]
    forms = query_variants(" ".join(naming)) if naming else []
    if forms:
        scored = sorted(((score(forms, aliases_for(o)), o) for o in options), reverse=True)
        if scored[0][0] >= 0.75 and (len(scored) == 1 or scored[1][0] < scored[0][0] - 0.05):
            return scored[0][1]
    if 0 < len(words) <= 4:
        for w in words:
            if w in _ORDINALS and abs(_ORDINALS[w]) <= len(options) - (0 if _ORDINALS[w] < 0 else 1):
                return options[_ORDINALS[w]]
    return None


class ConversationEngine:
    def __init__(self, kernel: ActionKernel, planner: Planner, *, composer: ResponseComposer | None = None,
                 history: HistoryStore | None = None, resources: ResourceManager | None = None,
                 memory: MemoryStore | None = None, router: FastPathRouter | None = None,
                 max_actions: int = 3) -> None:
        self.kernel = kernel
        self.planner = planner
        self.router = router
        self.composer = composer or ResponseComposer()
        self.history = history or InMemoryHistory()
        self.resources = resources
        self.memory = memory
        self.max_actions = max_actions
        # Keyed by (conversation, device): only the device that was asked can answer by voice/text.
        self._pending: dict[tuple[str, str | None], _PendingConfirmation] = {}
        self._choices: dict[tuple[str, str | None], _PendingChoice] = {}

    def handle(self, user: UserInput, emit: Emit) -> None:
        text = strip_wake_word(user.text)
        events: list[TurnEvent] = []

        def out(event: TurnEvent) -> None:
            events.append(event)
            emit(event)

        pending = self._pending.pop((user.conversation_id, user.device), None)
        if pending is not None:
            answer = parse_confirmation(text)
            self._answer(pending, answer is True, user, out)
            if answer is not None:
                self._record(user, text, events)
                return
            # Not a yes/no: the pending action was cancelled above; treat this as a new request.
        choice = self._choices.pop((user.conversation_id, user.device), None)
        if choice is not None:
            picked = pick_option(text, choice.options)
            if picked is not None:
                self._run(ActionStep(choice.step.tool, {**choice.step.args, choice.param: picked}),
                          user, choice.language, out)
                self._record(user, text, events)
                return
            if parse_confirmation(text) is False:
                out(Say(SayKind.RESULT, "Okay, I won't start anything.", choice.language))
                self._record(user, text, events)
                return
            # Anything else: a new request.
        if text:
            self._converse(text, user, out)
        self._record(user, text, events)

    def answer(self, confirmation_id: str, approved: bool, user: UserInput, emit: Emit) -> None:
        """Answer a confirmation by id (e.g. a button in the HUD or phone app)."""
        key = (user.conversation_id, user.device)
        pending = self._pending.get(key)
        if pending is not None and pending.confirmation_id == confirmation_id:
            del self._pending[key]
            self._answer(pending, approved, user, emit)
            return
        with self._busy(ResourceState.EXECUTING):
            outcome = self.kernel.resolve(confirmation_id, approved, device=user.device)
        self._report(outcome, "?", user, "en", emit)

    # --- internals ---------------------------------------------------------------

    def _converse(self, text: str, user: UserInput, out: Emit) -> None:
        history = self.history.recent(user.conversation_id)
        # Fast path first: trivial requests never load a language model.
        plan = self.router.route(text) if self.router is not None else None
        if plan is None:
            # Private memories are never put into prompts automatically.
            memories = self.memory.recall(text) if self.memory is not None else ()
            with self._busy(ResourceState.THINKING):
                try:
                    plan = self.planner.plan(text, history=history, memories=memories)
                except ProviderUnavailable as exc:
                    out(Say(SayKind.ERROR, f"My AI model isn't available right now ({exc}). "
                                           "Tools still work from the console with /run."))
                    return
                except PlanError as exc:
                    log.warning("planner output rejected: %s", exc)
                    out(Say(SayKind.ERROR, "Sorry, I didn't get that. Could you say it another way?"))
                    return

        if not plan.actions:
            verified_so_far = " ".join(m.content for m in history if m.role == "assistant")
            say, rewritten = self.composer.guard(plan.say, language=plan.language, context=verified_so_far)
            if rewritten:
                log.warning("replaced an unverified model reply: %r", plan.say)
            if say:
                out(Say(SayKind.REPLY, say, plan.language))
            return

        # With actions, the model's own words are not spoken: they could claim a result.
        if plan.acknowledge:
            out(Say(SayKind.REPLY, self.composer.acknowledge(plan.language, text), plan.language))
        steps = plan.actions[:self.max_actions]
        from_model = plan.acknowledge  # fast-path plans come from fixed rules, not the model
        for index, step in enumerate(steps):
            if self._run(step, user, plan.language, out, text=text if from_model else ""):
                if index < len(steps) - 1:
                    out(Say(SayKind.REPLY, "I'll wait for your answer before doing anything else.", plan.language))
                break

    def _ungrounded_launch(self, step: ActionStep, text: str) -> str | None:
        """A launch_app the model proposed for an app the user never mentioned: its display name, else None.

        Seen in evaluation: "wo browser wali baat bhool ja" (forget the browser thing) -> launch Steam.
        """
        if step.tool not in _GROUNDED_TOOLS or not isinstance(step.args.get("name"), str) or self.router is None \
                or self.router.apps is None:
            return None
        from nexus.system.apps.names import normalize, query_variants, score

        res = self.router.apps.resolve(step.args["name"])
        if res.match is None:
            return None  # nothing would launch anyway; the tool reports it
        said = normalize(text).split()
        # Grounded if any of the app's names appears in what the user said.
        for n in range(1, min(5, len(said)) + 1):
            for i in range(len(said) - n + 1):
                if score(query_variants(" ".join(said[i:i + n])), res.match.aliases) >= 0.85:
                    return None
        return res.match.display_name

    def _installed_app_instead_of_website(self, step: ActionStep, text: str) -> ActionStep:
        """The user named an installed app but the model chose a website: start the app.

        Seen in evaluation: "Chrome khol de re" -> open_url google.com/chrome. An installed
        application request must never become a website request.
        """
        if step.tool != "open_url" or self.router is None or self.router.apps is None:
            return step
        from nexus.system.apps.names import normalize, query_variants, score
        from nexus.system.apps.records import AppKind

        low = text.lower()
        if re.search(r"\.(?:com|org|net|in|io|dev|co|gov|edu|tv|me|ai|app|gg)\b|\b(?:website|site|web|online|"
                     r"page|link|url|search|google\s+kar)\b|https?://", low):
            return step  # the user asked for a website
        said = normalize(text).split()
        for n in range(min(4, len(said)), 0, -1):
            for i in range(len(said) - n + 1):
                phrase = " ".join(said[i:i + n])
                res = self.router.apps.resolve(phrase)
                if res.match is not None and res.match.kind in (AppKind.APP, AppKind.GAME) \
                        and score(query_variants(phrase), res.match.aliases) >= 0.95:
                    log.info("model chose a website for installed app %s; launching the app", res.match.display_name)
                    return ActionStep("launch_app", {"name": res.match.display_name})
        return step

    def _run(self, step: ActionStep, user: UserInput, language: str, out: Emit, *, text: str = "") -> bool:
        """Run one proposed action. Returns True if it is waiting for confirmation or a choice."""
        if text:
            step = self._installed_app_instead_of_website(step, text)
        ungrounded = self._ungrounded_launch(step, text) if text else None
        if ungrounded is not None:
            self._choices[(user.conversation_id, user.device)] = _PendingChoice(step, "name", (ungrounded,), language)
            out(Say(SayKind.CLARIFY, f"Should I {_GROUNDED_TOOLS[step.tool]} {ungrounded}? (yes/no)", language))
            return True
        request = ToolRequest(step.tool, step.args, source="ai", device=user.device)
        if self.kernel.preview(request).verdict is Verdict.ALLOW:
            self._announce(step, language, out)
        with self._busy(ResourceState.EXECUTING):
            outcome = self.kernel.submit(request)
        return self._report(outcome, step.tool, user, language, out, step=step)

    def _answer(self, pending: _PendingConfirmation, approved: bool, user: UserInput, out: Emit) -> None:
        if approved:
            self._announce(pending.step, pending.language, out)
        with self._busy(ResourceState.EXECUTING):
            outcome = self.kernel.resolve(pending.confirmation_id, approved, device=user.device)
        self._report(outcome, pending.step.tool, user, pending.language, out, step=pending.step)

    def _announce(self, step: ActionStep, language: str, out: Emit) -> None:
        tool = self.kernel.registry.get(step.tool)
        if tool is None:
            return
        try:
            args = tool.validate(step.args)
        except ValueError:
            return
        text = self.composer.announce(tool, args)
        if text:
            out(Say(SayKind.ANNOUNCE, text, language))

    def _report(self, outcome: Outcome, tool: str, user: UserInput, language: str, out: Emit,
                *, step: ActionStep | None = None) -> bool:
        verified = outcome.result.verified if outcome.result is not None else None
        out(ActionReport(tool, outcome.status, verified, outcome.message, outcome.request_id))
        if outcome.status is Status.NEEDS_CONFIRMATION and outcome.confirmation_id:
            if step is not None:
                self._pending[(user.conversation_id, user.device)] = _PendingConfirmation(
                    outcome.confirmation_id, step, language)
            prompt = self.composer.confirmation_prompt(outcome, language)
            out(Say(SayKind.CONFIRM, prompt, language))
            out(ConfirmationRequest(outcome.confirmation_id, prompt))
            return True
        if outcome.status is Status.NEEDS_CLARIFICATION and outcome.result is not None and step is not None:
            param = outcome.result.data.get("option_param") or next(
                (k for k, v in step.args.items() if isinstance(v, str)), None)
            if param is not None:
                self._choices[(user.conversation_id, user.device)] = _PendingChoice(
                    step, param, outcome.result.options, language)
            out(Say(SayKind.CLARIFY, outcome.message, language))
            return True
        out(self.composer.result(outcome, language))
        return False

    def _busy(self, state: ResourceState):
        return self.resources.busy(state) if self.resources is not None else nullcontext()

    def _record(self, user: UserInput, text: str, events: Sequence[TurnEvent]) -> None:
        if not text:
            return
        parts = []
        for e in events:
            if isinstance(e, Say):
                parts.append(e.text)
            elif isinstance(e, ActionReport):
                parts.append(f"[{e.tool}: {e.status.value}]")
        self.history.append(user.conversation_id, ChatMessage("user", text))
        if parts:
            self.history.append(user.conversation_id, ChatMessage("assistant", " ".join(parts)[:1500]))
