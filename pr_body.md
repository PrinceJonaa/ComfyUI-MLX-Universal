## What was broken
- `generate_text_embedding` in `runtime/embedding_processing.py` assumed tokenizers always successfully execute `encode(text, return_tensors='mlx')`. For standard HuggingFace tokenizers, this failed silently or threw errors.
- `execute_text_generation` and `execute_image_description` in `runtime/generate_processing.py` accessed `mlx_model.processor` without checking for `NoneType` safety, exposing type check errors and runtime crashes on some VLMs.
- `MLXKokoroTTS.generate_audio` and `MLXWhisperTranscribe.transcribe` in `nodes/audio_nodes.py` contained massive logic implementations within the UI nodes, violating the strict UI/MLX separation protocol.
- Assorted exception formatting across `diffusion_processing.py`, `bridge.py`, and `video_processing.py` did not match the strict UX format `<WHAT was expected> + <WHAT was actually found/happened> + <WHAT the user should do next>`.

## How it was verified
- Automated linters and type checkers passed.
- Unit tests (`make test`) verified regressions were not introduced.
- Refactored `audio_processing.py` executes successfully.
- Added explicit cache clear logic in chunk loops to ensure Apple Silicon Unified memory is conserved.

## Issues Resolved
- Enforces strict UI/MLX separation architecture in Audio nodes.
- Improves code health and handles silent/ambiguous exceptions via standard UX messaging.
