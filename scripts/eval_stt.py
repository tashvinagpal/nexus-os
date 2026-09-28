"""Speech-to-text accuracy on synthetic speech (Windows' offline voices), per model.

    .venv\\Scripts\\python.exe scripts\\eval_stt.py            # base and small

Only English voices exist on this PC, so this measures English, and Hinglish
*read by an English voice* (a rough stand-in, not a real Hinglish speaker).
Hindi, Punjabi and Haryanvi need real speakers: scripts/voice_check.py with a
microphone. For each phrase: what was written, whether it matches, which
language was detected, and whether the fast path would handle it without a
language model. Nothing is executed. Results go to var/eval/stt_eval.jsonl.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexus.ai.conversation.engine import strip_wake_word  # noqa: E402
from nexus.ai.router import FastPathRouter  # noqa: E402
from nexus.ai.voice.whisper import WhisperSTT  # noqa: E402
from nexus.core.config import PROJECT_ROOT  # noqa: E402
from nexus.system.apps.service import AppService  # noqa: E402
from nexus.system.platform import current  # noqa: E402
from tests.voice_fixtures import read_wav, synthesize  # noqa: E402

CASES = [  # (group, text spoken, fast-path tool expected or None)
    ("en", "What time is it?", "get_datetime"),
    ("en", "Open my downloads folder.", "open_folder"),
    ("en", "Open Chrome.", "launch_app"),
    ("en", "Nexus, launch GTA five.", "launch_app"),
    ("en", "Close Chrome.", "close_app"),
    ("en", "Go to YouTube.", "open_url"),
    ("en", "How much RAM is free?", "get_system_info"),
    ("en", "Remember that my favourite game is Rocket League.", "remember"),
    ("en", "What do you remember about me?", "list_memories"),
    ("en", "Yes.", None),
    ("en", "No, cancel that.", None),
    ("en", "I'm bored, what should I play?", None),
    ("hinglish", "Chrome khol do.", "launch_app"),
    ("hinglish", "GTA chala de.", "launch_app"),
    ("hinglish", "Kitne baje hain?", "get_datetime"),
    ("hinglish", "Downloads folder kholo.", "open_folder"),
    ("hinglish", "Haan, kar do.", None),
    ("hinglish", "Nahi, rehne do.", None),
]


def norm(text: str) -> str:
    text = text.lower().replace("favourite", "favorite").replace("five", "5").replace("gta 5", "gta v")
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text).split())


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    models = sys.argv[1:] or ["base", "small"]
    voices = ("Microsoft David Desktop", "Microsoft Zira Desktop")
    with tempfile.TemporaryDirectory() as tmp:
        wavs = synthesize({str(i): text for i, (_, text, _) in enumerate(CASES)}, tmp, voices=voices)
        clips = {key: read_wav(path) for key, path in wavs.items()}
    router = FastPathRouter(apps=AppService(current()))  # the installed apps, as NEXUS sees them (read-only)
    rows = []
    for model in models:
        stt = WhisperSTT(model, ram_mb=1, threads=6)
        stt.load()
        try:
            for (voice, i), clip in sorted(clips.items(), key=lambda kv: (kv[0][0], int(kv[0][1]))):
                group, text, tool = CASES[int(i)]
                t = stt.transcribe(clip)
                routed = router.route(strip_wake_word(t.text))
                got_tool = routed.actions[0].tool if routed and routed.actions else None
                rows.append({"model": model, "voice": voice, "group": group, "said": text, "heard": t.text,
                             "exact": norm(t.text) == norm(text), "language": t.language,
                             "confidence": round(t.confidence or 0, 2), "seconds": t.seconds,
                             "fast_path_expected": tool, "fast_path_got": got_tool})
        finally:
            stt.unload()
    for model in models:
        print(f"\n== whisper-{model} ==")
        for group in ("en", "hinglish"):
            sel = [r for r in rows if r["model"] == model and r["group"] == group]
            exact = sum(r["exact"] for r in sel)
            routed = [r for r in sel if r["fast_path_expected"]]
            ok = sum(r["fast_path_got"] == r["fast_path_expected"] for r in routed)
            print(f"  {group:<9} exact words {exact}/{len(sel)}   fast path correct {ok}/{len(routed)}   "
                  f"median {sorted(r['seconds'] for r in sel)[len(sel) // 2]:.2f} s")
            for r in sel:
                if not r["exact"]:
                    print(f"     {r['voice']:<5} said {r['said']!r:<52} heard {r['heard']!r} [{r['language']}]")
    out = PROJECT_ROOT / "var" / "eval" / "stt_eval.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    with out.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"at": stamp, **r}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
