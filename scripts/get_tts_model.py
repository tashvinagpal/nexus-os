"""Download the Kokoro text-to-speech model into models/tts/ (the only way NEXUS gets one).

    .venv\\Scripts\\python.exe scripts\\get_tts_model.py            # int8 model + voices (114 MB), asks first
    .venv\\Scripts\\python.exe scripts\\get_tts_model.py fp32 --yes # full-precision model (310 MB)

Files come from the kokoro-onnx project's release (Kokoro-82M, Apache-2.0).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexus.ai.voice.kokoro import KOKORO_FILES, MODELS_DIR  # noqa: E402

_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variant", nargs="?", default="int8", choices=sorted(k for k in KOKORO_FILES if k != "voices"))
    parser.add_argument("--yes", action="store_true", help="don't ask before downloading")
    args = parser.parse_args()
    wanted = [KOKORO_FILES[args.variant], KOKORO_FILES["voices"]]
    missing = [(name, mb) for name, mb in wanted if not (MODELS_DIR / name).is_file()]
    if not missing:
        print(f"Already in {MODELS_DIR}")
        return 0
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    total = sum(mb for _, mb in missing)
    free_mb = shutil.disk_usage(MODELS_DIR).free // 2**20
    for name, mb in missing:
        print(f"  {name}: about {mb} MB")
    print(f"Total about {total} MB from {_BASE} into {MODELS_DIR} ({free_mb:,} MB free)")
    if total * 2 > free_mb:
        print("Not enough free disk space.")
        return 1
    if not args.yes and input("Download? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Nothing downloaded.")
        return 1
    for name, _ in missing:
        part = MODELS_DIR / (name + ".part")
        with urllib.request.urlopen(_BASE + name, timeout=60) as response, part.open("wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
        part.replace(MODELS_DIR / name)
        print(f"  {name}: {(MODELS_DIR / name).stat().st_size // 2**20} MB")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
