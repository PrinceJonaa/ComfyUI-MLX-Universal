## Architectural Thesis

The `nodes/audio_nodes.py` UI wrappers contained hardcoded MLX import logic, memory evaluation commands, and PyTorch bridging inside the node generation methods `transcribe` and `generate_audio`. This violates the strict architectural invariant of separating UI logic from the MLX execution runtime. This PR retires this debt by introducing a new `execute_tts_generation` abstraction in the Runtime Substrate and migrating the nodes to delegate their workloads there, eliminating duplicated boilerplate, and consolidating the tensor bridge logic for the audio modality.

## Debt Location

- `nodes/audio_nodes.py` (line 116): The `MLXKokoroTTS.generate_audio` method contained leaking MLX processing abstractions and PyTorch bridging code.
- `nodes/audio_nodes.py` (line 35): The `MLXWhisperTranscribe.transcribe` method was violating the Runtime Substrate separation by explicitly performing MLX-whisper generation and file handling logic.

## What Changed

- Abstracted Kokoro TTS generation out of `MLXKokoroTTS.generate_audio` and migrated it to a new `execute_tts_generation` utility inside `runtime/audio_processing.py`.
- Updated `MLXKokoroTTS` to cleanly delegate its parameter payload directly to `execute_tts_generation`.
- Updated `MLXWhisperTranscribe` to cleanly delegate its payload to the pre-existing `execute_audio_transcription` function within `runtime/audio_processing.py`.
- Removed unused file-handling dependencies (`os`, `tempfile`, `soundfile`) from `nodes/audio_nodes.py`.

## What Was Not Changed

- The node schemas (`INPUT_TYPES`, `RETURN_TYPES`), signature names (`MLXKokoroTTS`, `MLXWhisperTranscribe`), and output tuple shapes were completely preserved. No saved workflows or node logic interfaces were broken during this migration.

## Backward Compatibility

- Successfully validated compatibility by ensuring the extracted `execute_tts_generation` and the updated nodes pass static validation and maintain standard output shapes (a tuple containing a dictionary representation of the audio).
- Tested against Mock testing environment ensuring that `make verify` and tests run green on standard interfaces.

## Rejected Alternatives

- A rejected alternative was migrating the audio parsing functions to standard UI nodes. This was ignored because memory management, such as evaluation and dictionary creation, fundamentally belongs inside the Runtime Substrate to avoid UI-blocking MLX execution faults.

## Follow-On Candidates

- The SAM3 Predictor in `nodes/sam_nodes.py` still contains minor logic mapping that could be pushed directly into the `process_sam3_result` processor inside the runtime folder for better UX separation.
