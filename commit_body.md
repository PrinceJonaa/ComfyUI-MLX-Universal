## Architectural Thesis

The `MLXKokoroTTS` node was violating the core architectural constraint defined in `CONTRIBUTING.md`: strict separation of UI logic and MLX execution. The generation loop, audio chunking, MLX tensor evaluation, and conversion back to PyTorch were all happening directly inside the ComfyUI frontend wrapper. This created structural debt (Leaking abstraction / Duplicated boilerplate) where the UI layer had excessive knowledge of MLX internals. Extracting this into `execute_kokoro_tts` in `runtime/audio_processing.py` retires this debt by shielding the UI from the execution runtime, making testing, tracing, and future modifications safer and easier.

## Debt Location

`nodes/audio_nodes.py`, inside the `MLXKokoroTTS.generate_audio` method (approx. lines 95-128).

## What Changed

- **Extracted**: The MLX execution logic, pipeline iteration, array concatenation, lazy evaluation boundaries, and tensor bridging were moved to a new function `execute_kokoro_tts` in `runtime/audio_processing.py`.
- **Consolidated**: The `MLXKokoroTTS.generate_audio` method in `nodes/audio_nodes.py` was reduced to a single line that imports and calls `execute_kokoro_tts`, preserving its original tuple return signature.

## What Was Not Changed

- **Public Interfaces**: The `INPUT_TYPES` and `RETURN_TYPES` of `MLXKokoroTTS` remain completely unchanged.
- **Serialization**: Saved workflows utilizing the "MLX Generate Audio (Kokoro)" node will deserialize identically.
- **Registrations**: `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` were untouched.

## Backward Compatibility

- [x] Verified `nodes/audio_nodes.py` module attributes programmatically (confirmed identical node signature).
- [x] Confirmed `make verify` tests complete fully with the new runtime extraction structure, showing tests exercise the mocked MLX substrate successfully.
- [x] Checked that no new `runtime/` internals were leaked into `nodes/`.

## Rejected Alternatives

I considered also extracting `MLXSAM3Predictor` logic which has a similar pattern. However, the strict single-pass Refactor scope rule limits changes to one specific abstraction extraction. Bundling SAM3 would increase regression risk and violate the PR delivery specification.

## Follow-On Candidates

- **`MLXSAM3Predictor` Extraction**: The SAM3 prediction node still handles bounding box conversions and PIL Image processing inside the ComfyUI node file. This should be deferred to a future Structural Refactor pass.
