"""Entry point: `python -m nexus` or `nexus`."""

from __future__ import annotations

import argparse
import logging
import sys

from nexus import __version__
from nexus.console import Console
from nexus.core.config import (ConfigError, load_ai_settings, load_app_settings, load_settings,
                               load_tts_settings, load_voice_settings)
from nexus.core.resources import ResourceManager
from nexus.core.runtime import create_runtime
from nexus.security.permissions import PolicyError
from nexus.system import platform as nexus_platform


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nexus", description="NEXUS OS console")
    parser.add_argument("--version", action="version", version=f"NEXUS {__version__}")
    parser.add_argument("--no-ai", action="store_true", help="start without the AI layer (tools only)")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]

    plat = nexus_platform.current()
    if plat.is_elevated():
        print("NEXUS will not run as Administrator. It is designed to run with normal user "
              "rights so that nothing it does can change protected parts of Windows.\n"
              "Open a normal (non-admin) terminal and try again.")
        return 1

    settings = load_settings()
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=settings.log_file, level=logging.INFO, encoding="utf-8",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        ai = load_ai_settings(settings.config_file)
        voice = load_voice_settings(settings.config_file)
        speech = load_tts_settings(settings.config_file)
        resources = ResourceManager(ram_reserve_mb=ai.ram_reserve_mb, vram_reserve_mb=ai.vram_reserve_mb)
        runtime = create_runtime(settings, plat, resources=resources,
                                 app_settings=load_app_settings(settings.config_file))
        runtime.apps.warm_up()               # discover installed apps in the background
        runtime.apps.start_game_watcher()    # game mode while a known game runs
        assistant = None
        if not args.no_ai:
            from nexus.core.assistant import create_assistant

            assistant = create_assistant(runtime, ai, voice=voice, speech=speech)
    except (PolicyError, ConfigError, RuntimeError) as exc:
        print(f"NEXUS did not start: {exc}")
        return 1
    logging.getLogger(__name__).info("NEXUS %s started", __version__)
    try:
        Console(runtime, assistant).run()
    finally:
        if assistant is not None:
            assistant.speaker.stop()  # (model worker processes end by themselves when NEXUS closes their pipe)
    return 0


if __name__ == "__main__":
    sys.exit(main())
