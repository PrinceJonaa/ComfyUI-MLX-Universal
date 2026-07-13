## Architectural Thesis

The `MLXKokoroTTS` UI node previously contained inline MLX processing logic, including array concatenation, memory evaluation (`mx.eval`), and tensor bridging. This violated the strict architectural invariant separating ComfyUI wrappers from background MLX execution. By extracting this generation loop into `execute_tts_generation` within the `runtime` layer, this refactor ensures consistent VRAM management, prevents memory deadlock cascades when bridging to PyTorch, and enforces structural hygiene across modalities.

## Debt Location

- `nodes/audio_nodes.py`: `MLXKokoroTTS.generate_audio` (Lines 87-111 prior to PR)

## What Changed

Extracted the generation logic from the Kokoro TTS node into `execute_tts_generation` in `runtime/audio_processing.py`.
The caller in `MLXKokoroTTS` now cleanly delegates to this function.

**Before:**
```python
    def generate_audio(self, text: str, voice: str, speed: float) -> tuple:
        from ..runtime.model_loader import load_kokoro_pipeline
        pipeline = load_kokoro_pipeline("prince-canuma/Kokoro-82M")
        ...
        final_audio_tensor = mlx_to_torch(final_audio).float()
        ...
        return (audio_out,)
```

**After:**
```python
    def generate_audio(self, text: str, voice: str, speed: float) -> tuple:
        from ..runtime.audio_processing import execute_tts_generation

        audio_out = execute_tts_generation(text, voice, speed)
        return (audio_out,)
```

## What Was Not Changed

- The public schema of `MLXKokoroTTS` (including `INPUT_TYPES` and `RETURN_TYPES`) remains identical.
- Saved user workflows using this node will deserialize perfectly.
- No new imports bleeding `mlx.core` were introduced into the UI layer.

## Backward Compatibility

- Ran `make test` locally. Confirmed the mock-based unit tests for `MLXKokoroTTS` pass.
- Verified that no changes to public ComfyUI node interfaces were made.
- Ran `make verify` to ensure linter and type-checker greenlights without unused imports.

## Rejected Alternatives

I considered extracting both Kokoro TTS and Whisper Transcribe into the runtime layer in a single pass. However, per the strict "One refactor per PR" and "Internal-only changes first" constraints, doing both simultaneously risks breaking the test suite through scope-creep. We defer Whisper migration to a dedicated PR.

## Follow-On Candidates

- **Deferred:** Migrate `MLXWhisperTranscribe.transcribe` to correctly delegate to the existing `execute_audio_transcription` runtime function. This was deferred to adhere strictly to the one-refactor-per-PR rule.
