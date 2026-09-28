"""Download the two shared openWakeWord feature networks into models/wake/ (~2.4 MB, Apache-2.0).

    .venv\\Scripts\\python.exe scripts\\get_wake_model.py
    .venv\\Scripts\\python.exe scripts\\get_wake_model.py --hey-jarvis   # + a ready-made "Hey Jarvis" model to try

They turn audio into speech features for every wake word; the "Nexus" model on top of them is
made on this PC by scripts/train_wake_model.py (no download). Files come from the openWakeWord
project's GitHub release and are checked against known SHA-256 hashes before they are used.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexus.core.config import PROJECT_ROOT  # noqa: E402

_BASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"
FILES = {
    "melspectrogram.onnx": "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
    "embedding_model.onnx": "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
}
HEY_JARVIS = {"hey_jarvis_v0.1.onnx": "94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb"}
FOLDER = PROJECT_ROOT / "models" / "wake"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(name: str, digest: str) -> bool:
    target = FOLDER / name
    if target.is_file() and sha256(target) == digest:
        print(f"  {name}: already here")
        return True
    part = target.with_suffix(".part")
    try:
        with urllib.request.urlopen(_BASE + name, timeout=60) as response, part.open("wb") as out:
            out.write(response.read())
    except OSError as exc:
        print(f"  {name}: download failed ({exc})")
        part.unlink(missing_ok=True)
        return False
    if sha256(part) != digest:
        print(f"  {name}: the downloaded file isn't the expected one - not used")
        part.unlink(missing_ok=True)
        return False
    part.replace(target)
    print(f"  {name}: {target.stat().st_size // 1024} KB")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hey-jarvis", action="store_true",
                        help='also get openWakeWord\'s "Hey Jarvis" model (set [wake] model = "hey_jarvis_v0.1")')
    args = parser.parse_args()
    FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"Into {FOLDER} from {_BASE}")
    files = {**FILES, **(HEY_JARVIS if args.hey_jarvis else {})}
    ok = all([fetch(name, digest) for name, digest in files.items()])
    if ok and not (FOLDER / "nexus.onnx").is_file():
        print("Next: make the \"Nexus\" model - .venv\\Scripts\\python.exe scripts\\train_wake_model.py")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
