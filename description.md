## What This Fixes

This PR removes friction points for first-time users by replacing confusing ValueError exceptions with clear, actionable error messages formatted as `<expected> + <found> + <user action>`. Additionally, it lowers video generation defaults to prevent immediate Out-of-Memory (OOM) failures on standard consumer hardware (like 16GB unified memory Macs), adds helpful tooltips to technical parameters, and syncs the documentation to reflect recently added capabilities.

## Error Message Changes

```python
# Before
raise ValueError("Draft model path cannot be empty.")
# After
raise ValueError("Expected a valid draft model path string but found an empty string. Please provide a valid Hugging Face repository ID or local directory path for the draft model.")

# Before
raise ValueError(f"Unsupported draft model type: {model_type}")
# After
raise ValueError(f"Expected a supported draft model type ('mlx-lm' or 'mlx-vlm') but found '{model_type}'. Please ensure your draft model type is set correctly in the node.")

# Before
raise ValueError(
    f"Expected a known resolved model type (e.g. 'mlx-lm', 'mlx-vlm', 'sam3') + Found '{resolved_type}' + Check the model path or force the model type manually"
)
# After
raise ValueError(
    f"Expected a known resolved model type (e.g., 'mlx-lm', 'mlx-vlm', 'sam3') but found '{resolved_type}'. Check the model path or try manually selecting the model type instead of 'auto'."
)

# Before
raise ValueError("Kokoro TTS generated no audio for the given text.")
# After
raise ValueError("Expected Kokoro TTS to generate an audio waveform but found none. This usually means the input text was empty or contained unprocessable characters. Try simplifying your text prompt.")

# Before
raise ValueError(
    "Expected a valid MLX conditioning dictionary from an MLX Load Flux Model node + Invalid or missing conditioning input + Ensure the MLX Load Flux Model node is properly connected"
)
# After
raise ValueError(
    "Expected a valid MLX conditioning dictionary from an MLX Load Flux Model node but found an invalid or missing conditioning input. Ensure the MLX Load Flux Model node is properly connected."
)
```

## Default Changes

- `num_frames` in `MLXVideoGenerator`: **16 -> 8**. Generating 16 frames of video via CLI subprocess is highly likely to trigger an OS-level hard-swap or crash on machines with 16GB unified memory if the user is unaware of the cost. Lowering to 8 ensures a higher likelihood of first-run success.

## Documentation Updates

- Updated the Core Capabilities table in `README.md` to explicitly mention `Kokoro` TTS integration under Audio, and added a new row for `Embeddings`.
- Updated the Architecture Map mermaid diagram to include previously omitted frontend nodes: `MLX Load Draft Model`, `MLX Batch Understand Image`, `MLX Generate Audio (Kokoro)`, and `MLX Generate Text Embedding`.

## Explicitly Out of Scope

- Refactoring the underlying string parsing or generation logic causing the missing audio / unexpected model type errors is out of scope. These are strictly UI and UX debt optimizations.

## Verification

- Verified all parameter keys remain unchanged (only defaults and `tooltip` metadata dictionaries were updated) to ensure backward compatibility with saved user workflows.
- Diff confirms no execution logic or tensor math was altered.
- All unit tests passed via `python3 -m unittest discover tests` and static verification passed via `make verify`.
