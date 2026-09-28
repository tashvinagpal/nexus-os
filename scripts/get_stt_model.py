"""Download a faster-whisper speech-to-text model into models/stt/ (the only way NEXUS gets one).

    .venv\\Scripts\\python.exe scripts\\get_stt_model.py small          # asks before downloading
    .venv\\Scripts\\python.exe scripts\\get_stt_model.py base --yes

Shows the download size first. Files come from the model's Hugging Face
repository (CTranslate2 conversions of OpenAI Whisper, MIT licence).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nexus.ai.voice.whisper import MODELS_DIR, WHISPER_MODELS, model_folder  # noqa: E402

_FILES = ["config.json", "model.bin", "tokenizer.json", "vocabulary.*", "preprocessor_config.json"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=sorted(WHISPER_MODELS))
    parser.add_argument("--yes", action="store_true", help="don't ask before downloading")
    args = parser.parse_args()
    repo, size_mb = WHISPER_MODELS[args.model]
    folder = model_folder(args.model)
    if (folder / "model.bin").is_file():
        print(f"{args.model} is already in {folder}")
        return 0
    free_mb = shutil.disk_usage(MODELS_DIR.parent if MODELS_DIR.parent.exists() else Path.cwd()).free // 2**20
    print(f"Whisper {args.model}: about {size_mb} MB from https://huggingface.co/{repo}")
    print(f"into {folder}  ({free_mb:,} MB free on that drive)")
    if size_mb * 2 > free_mb:
        print("Not enough free disk space.")
        return 1
    if not args.yes and input("Download? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Nothing downloaded.")
        return 1
    from huggingface_hub import snapshot_download

    folder.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, local_dir=str(folder), allow_patterns=_FILES)
    total = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) // 2**20
    print(f"Done: {total} MB in {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
