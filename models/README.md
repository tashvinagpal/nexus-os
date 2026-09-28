# models/

Local AI model files (speech recognition, wake word, TTS, vision) live here.
They are large, so this folder is git-ignored except for this file.

LLMs are managed by Ollama (stored in `%USERPROFILE%\.ollama\models`), not here.

Nothing is downloaded automatically. Each model is added deliberately, after
its size, VRAM/RAM needs, and expected speed have been written down in
`docs/architecture.md`.
