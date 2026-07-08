[Refactor] Audio Nodes: Extract MLX execution logic into Runtime Substrate

## Architectural Thesis

The MLX execution logic for audio transcription and generation (Whisper and Kokoro models) was improperly residing directly within the UI nodes (`MLXWhisperTranscribe` and `MLXKokoroTTS`), violating the Runtime Substrate separation invariant. This leaking abstraction created a risk where modifying MLX generation or tensor slicing logic required coordinating changes inside the ComfyUI nodes, potentially leading to instability or crashing the ComfyUI process directly on failure. This refactor specifically retires that debt by extracting the execution and logic out of the `nodes/` module and properly delegating it to the background `runtime/audio_processing.py` handlers, adhering strictly to the separation of UI logic and MLX execution.

## Debt Location

Before this PR, the debt lived precisely in `nodes/audio_nodes.py`:
- `MLXWhisperTranscribe.transcribe` (lines 29-57)
- `MLXKokoroTTS.generate_audio` (lines 92-127)

## What Changed

- Introduced `execute_kokoro_tts` in `runtime/audio_processing.py` to handle MLX-specific generation, caching, and tensor conversion for Kokoro TTS logic.
- Migrated the existing Whispher logic in `MLXWhisperTranscribe.transcribe` to delegate exactly to the pre-existing, unused `execute_audio_transcription` abstraction in `runtime/audio_processing.py`.
- Migrated `MLXKokoroTTS.generate_audio` to delegate exactly to the newly created `execute_kokoro_tts` abstraction.
- Removed unused imports (`tempfile`, `soundfile`, `os`) from `nodes/audio_nodes.py`.
- Updated unit tests (`tests/test_audio_nodes.py` and `tests/test_runtime_audio.py`) to appropriately assert calls to the extracted boundary methods rather than the core `mlx` framework implementations.

## What Was Not Changed

- The public node interfaces, node schemas (`INPUT_TYPES`, `RETURN_TYPES`), return formats, and internal registry names were explicitly verified to be unchanged. No compatibility breaks for deserialization.
- The data payload and the core validation step `if not isinstance(audio, dict) ...` inside `MLXWhisperTranscribe.transcribe` remains untouched, preventing bad datatypes before attempting bridging.

## Backward Compatibility

- Verified deserialization of schemas conceptually remains valid since `INPUT_TYPES` and `RETURN_TYPES` remain entirely unmodified.
- Confirmed that the node types that were touched still register correctly with ComfyUI's mapping dictionary (`NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS`).
- Confirmed unit tests all pass locally via `make verify` and no `nodes/` import depends on inner internals.

## Rejected Alternatives

The next most viable alternative was to refactor `nodes/sam_nodes.py` as it also violates the Runtime Substrate rule by interacting directly with `Sam3Predictor` and `mlx_model.processor`. This was rejected for this specific cycle because the Audio node refactoring provided a higher immediate impact (addressing two independent models vs one) and lower risk, as `execute_audio_transcription` was already partially built out and only required final wiring and extraction of the Kokoro equivalent.

## Follow-On Candidates

- The SAM 3 `MLXSAM3Predictor.predict` node in `nodes/sam_nodes.py` continues to violate the Runtime Substrate boundary by interacting directly with the `mlx_vlm.models.sam3.generate.Sam3Predictor` module instead of delegating to `runtime/sam_processing.py`. This was deferred to maintain the single-refactor-per-PR constraint.
