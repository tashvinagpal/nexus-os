"""Choosing the speech-to-text model for each utterance, resource-aware.

Same rules as the chat models (nexus/ai/providers/selection.py): best option
first, skipped if it isn't installed, if cloud AI is off or there is no
network (for LAN/cloud providers), or if the ResourceManager won't admit it
right now. On top of that:

- While a game runs (GAME_MODE) only options marked `game_mode_ok` (light,
  CPU, few threads) are used, and an idle heavier one is unloaded at once.
- If a model fails to load (e.g. the GPU libraries are missing), it is skipped
  for a while and the next option is used - so GPU failures fall back to CPU.
- The model starts loading as soon as listening starts (`prepare`), so its
  load time is hidden behind the time you spend speaking.
"""

from __future__ import annotations

import logging
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Sequence

from nexus.ai.providers.selection import ProviderOption, SelectionContext, select_provider
from nexus.ai.voice.base import AudioClip, STTUnavailable, Transcript
from nexus.core.resources import ResourceManager, ResourceUnavailable

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class STTOption(ProviderOption):
    game_mode_ok: bool = False  # light enough to run next to a game


class HybridSTT:
    def __init__(self, options: Sequence[STTOption], resources: ResourceManager,
                 context: Callable[[], SelectionContext]) -> None:
        self.options = list(options)
        self.resources = resources
        self.context = context
        self.last_choice: str | None = None
        for option in self.options:
            if option.spec is not None:
                resources.register(option.provider)  # type: ignore[arg-type]  # local providers are ModelBackends

    def choose(self) -> tuple[STTOption | None, list[str]]:
        options, skipped = self.options, []
        if self.resources.game_mode:
            heavy = [o for o in options if not o.game_mode_ok]
            for option in heavy:
                skipped.append(f"{option.provider.info.name}: too heavy while a game runs")
                if option.spec is not None and self.resources.unload(option.spec.name):
                    log.info("unloaded %s for game mode", option.spec.name)
            options = [o for o in options if o.game_mode_ok]
        choice, more = select_provider(options, self.resources, self.context())
        for name, option in ((o.provider.info.name, o) for o in options):
            problem = getattr(option.provider, "problem", "")
            more = [f"{name}: {problem}" if s == f"{name}: not available" and problem else s for s in more]
        return choice, skipped + more  # type: ignore[return-value]

    def prepare(self) -> STTOption | None:
        """Pick a model and start loading it in the background. None (with no error) if none can run."""
        option, _ = self.choose()
        if option is None or option.spec is None or option.provider.is_loaded():  # type: ignore[attr-defined]
            return option
        name = option.spec.name

        def load() -> None:
            try:
                with self.resources.use(name):
                    pass
            except (ResourceUnavailable, STTUnavailable) as exc:
                log.info("preloading %s failed: %s", name, exc)
            except Exception:
                log.exception("preloading %s failed", name)

        threading.Thread(target=load, name="nexus-stt-preload", daemon=True).start()
        return option

    def transcribe(self, audio: AudioClip, *, language_hint: str | None = None,
                   cancel: threading.Event | None = None) -> Transcript:
        problems: list[str] = []
        for _ in range(len(self.options)):
            option, skipped = self.choose()
            if option is None:
                break
            name = option.provider.info.name
            hold = self.resources.use(option.spec.name) if option.spec else nullcontext()
            try:
                with hold:
                    transcript = option.provider.transcribe(audio, language_hint=language_hint,  # type: ignore[attr-defined]
                                                            cancel=cancel)
                self.last_choice = name
                return transcript
            except (ResourceUnavailable, STTUnavailable) as exc:  # couldn't load: the next option gets a turn
                problems.append(str(exc))
                log.info("speech model %s unavailable: %s", name, exc)
        _, skipped = self.choose()
        reasons = list(dict.fromkeys(problems + skipped)) or ["no speech-to-text model is configured"]
        raise STTUnavailable("I can't transcribe speech right now: " + "; ".join(reasons))
