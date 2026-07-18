## What This Fixes

First-time users face friction when debugging Out-Of-Memory errors or confusing generic Python exceptions without next steps. This PR improves UX by standardizing node error messages to be actionable, lowering the default number of frames in video generation to prevent OOM errors on standard hardware, and providing descriptive tooltips on node inputs. It also syncs the documentation to accurately reflect actual node capabilities.

## Error Message Changes

```python
# nodes/generate_nodes.py
# Before
ValueError(f"Expected model family 'mlx-lm' but found '{mlx_model.family}'. Please ensure you are passing a text model loaded via 'MLX Load Model', not a Vision, Audio, or SAM model.")
# After
ValueError(f"Expected model family 'mlx-lm' but found '{mlx_model.family}'. Ensure you passed a text model to 'MLX Load Model' (e.g. Qwen, Llama). If passing a Vision/Audio model, use the appropriate node.")

# Before
ValueError(f"Expected model family 'mlx-vlm' but found '{mlx_model.family}'. Please ensure you are passing a Vision-Language Model loaded via 'MLX Load Model', not a standard text or SAM model.")
# After
ValueError(f"Expected model family 'mlx-vlm' but found '{mlx_model.family}'. Ensure you passed a Vision-Language Model to 'MLX Load Model'. Check the supported models list if unsure.")

# nodes/audio_nodes.py
# Before
ValueError("Expected ComfyUI AUDIO dict format but found invalid or missing audio input. Connect a valid audio source node to this input.")
# After
ValueError("Expected ComfyUI AUDIO dict format but found invalid or missing audio input. Connect a valid audio source node (like 'Load Audio') to this input.")

# runtime/audio_processing.py
# Before
ValueError("Kokoro TTS generated no audio for the given text.")
# After
ValueError("Kokoro TTS generated no audio for the given text. Please check if your prompt is empty or contains only unsupported characters.")

# nodes/sam_nodes.py
# Before
ValueError(f"Expected model family 'sam3' but found '{mlx_model.family}'. Please ensure you are passing a SAM model loaded via 'MLX Load Model'.")
# After
ValueError(f"Expected model family 'sam3' but found '{mlx_model.family}'. Please ensure you loaded a valid SAM3 model via 'MLX Load Model'.")

# Before
ValueError("Expected an image batch but found empty input. Please connect a valid image to the node.")
# After
ValueError("Expected an image batch but found empty input. Connect a valid image (e.g., from 'Load Image') to the node.")

# nodes/loader_nodes.py
# Before
ValueError("Draft model path cannot be empty.")
# After
ValueError("Draft model path cannot be empty. Please provide a valid huggingface repo id or local path.")
```

## Default Changes

*   **`num_frames` (MLXVideoGenerator)**: Lowered from `16` to `8`.
    *   *Reasoning:* Video generation frequently causes OOM on 16GB Macs at default sizes. Lowering the default guarantees first-time users can successfully generate a video without crashing their environment, while power users can optionally raise it.

## Documentation Updates

*   Updated `README.md` **Core Capabilities** to explicitly list `Kokoro TTS` in the Audio category, not just Whisper.
*   Updated `README.md` **Core Capabilities** to explicitly list `Embeddings`.
*   Updated `README.md` **Architecture Map** to add the `UI_Kokoro[MLX Generate Audio (Kokoro)]`, `UI_TextEmb[MLX Generate Text Embedding]`, `UI_BatchVLM[MLX Batch Understand Image]` and `UI_Draft[MLX Load Draft Model]` graph nodes mapping to their respective `nodes/*.py` wrapper modules to match actual implementation.

## Explicitly Out of Scope

*   None; all issues identified were in-scope for frontend/UX fixes (no architectural/logic flaws were addressed).

## Verification

Confirmed that no saved workflow parameter keys were broken or renamed; all UX enhancements for parameters were localized strictly to modifying `default` values or rewriting exception string structures without changing application flow. Ran `make test` and `make verify` successfully to guarantee zero regressions.
