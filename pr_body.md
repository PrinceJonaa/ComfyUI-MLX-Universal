## Architectural Thesis

The MLX audio generation and transcription implementation details (lazy evaluation triggers, explicit tensor concatenations, and third-party pipeline instantiations) were directly embedded inside the ComfyUI node wrappers in `nodes/audio_nodes.py`. This created a leaking abstraction where the UI layer had to juggle core dependencies (like `mlx.core` and `mlx_whisper`) and perform heavy data transformations. By extracting `execute_tts_generation` and utilizing `execute_audio_transcription` inside `runtime/audio_processing.py`, we isolate these responsibilities strictly to the runtime substrate, completely satisfying the strict UI/MLX separation invariant while preventing future UI changes from accidentally breaking the audio generation pipeline.

## Debt Location

- `nodes/audio_nodes.py`: Execution functions inside `MLXWhisperTranscribe` (lines 35-71) and `MLXKokoroTTS` (lines 115-144).

## What Changed

- Introduced `execute_tts_generation` into `runtime/audio_processing.py` to handle the Kokoro model pipeline and tensor processing.
- Refactored `MLXKokoroTTS.generate_audio` to strictly delegate its execution to `execute_tts_generation`.
- Refactored `MLXWhisperTranscribe.transcribe` to delegate its execution to the existing `execute_audio_transcription`.
- Stripped all direct MLX manipulation imports (`mlx.core`, `mlx_whisper`, `numpy`, `torch`, `tempfile`, `soundfile`, `os`) out of `nodes/audio_nodes.py`, converting it into a pure dictionary-passing frontend wrapper.

## What Was Not Changed

- All ComfyUI node definitions (`INPUT_TYPES`, `RETURN_TYPES`, `RETURN_NAMES`) remain 100% identical.
- The public signatures, categories, and parameter keys are untouched. Existing saved user workflows will successfully deserialize and run without any modification.

## Backward Compatibility

- Verified backwards compatibility by running the suite of global tests (`python3 -m unittest discover tests`) and node tests (`python3 -m unittest discover nodes/tests`), successfully validating the modified routing.
- Ran a local python script to verify that `import nodes.audio_nodes` executes correctly without raising import errors, confirming successful node initialization.
- Passed all syntax, complexity, and formatting checks via `make verify`.

## Rejected Alternatives

- **Extracting input validation (e.g., checking for 'waveform') into `audio_processing.py`**: This was rejected because the ComfyUI-specific `AUDIO` dict format validation is arguably a concern of the UI bridge layer. Keeping the initial schema check in the node wrapper prevents the backend runtime from being overly coupled to ComfyUI's specific idiosyncratic data types.

## Follow-On Candidates

- Consider migrating the remaining input validation inside `MLXWhisperTranscribe` down to the bridge layer to make the UI node even thinner, though this requires careful type handling to avoid coupling the runtime to ComfyUI dict specs.
