## Architectural Thesis

The `nodes/audio_nodes.py` file suffered from a "Leaking abstraction" structural debt where execution loops (specifically TTS chunk processing, evaluating lazy tensors, and model loading) were tightly coupled to the UI node wrappers. This violated the strict separation of UI logic and MLX execution invariant described in `CONTRIBUTING.md`, making the codebase harder to maintain and test, and increasing the risk of ComfyUI startup crashes if environments are slightly off. This refactor cleanly extracts the execution loop into the `runtime/audio_processing.py` substrate.

## Debt Location

- `nodes/audio_nodes.py`, inside `MLXKokoroTTS.generate_audio` and `MLXWhisperTranscribe.transcribe`.

## What Changed

- Extracted the TTS array processing and Whisper inference loop into `execute_kokoro_tts` and `execute_audio_transcription` respectively inside `runtime/audio_processing.py`.
- Updated `nodes/audio_nodes.py` methods to strictly delegate to the `runtime/audio_processing.py` functions without directly referencing OS logic, `mlx.core` or the pipeline objects.
- Updated `tests/test_audio_nodes.py` to assert the delegation behavior directly using the runtime abstractions instead of duplicating implementation mocks.
- Implemented `tests/test_runtime_audio.py` for testing the `execute_kokoro_tts` runtime abstraction.

## What Was Not Changed

- The public node interfaces, node schemas (`INPUT_TYPES`, `RETURN_TYPES`), and registration names in `nodes/audio_nodes.py` remain entirely unchanged. Deserialization of existing user workflows works identically.
- No tensor types or dictionary schemas were modified.

## Backward Compatibility

- Verified by manually reviewing `nodes/audio_nodes.py` after editing to guarantee exact match in `INPUT_TYPES` and `RETURN_TYPES`.
- All global tests and specific node tests ran and passed (via `make test` and `make verify`), confirming no regressions in inputs or outputs.

## Rejected Alternatives

- Refactoring multiple files simultaneously (e.g., video and audio in one PR). This was rejected to strictly adhere to the "One refactor per PR" boundary, minimizing regression risk and keeping the PR scoped correctly.

## Follow-On Candidates

- The `nodes/video_nodes.py` file should also be reviewed in a future cycle. Although it uses a delegated structure, its wrapper could potentially be simplified.
- Error handling inside `MLXKokoroTTS` could be more symmetrical across different system constraints (e.g. out of memory, or invalid inputs missing TTS chunks). Deferred for now to only focus on the structural decoupling.
