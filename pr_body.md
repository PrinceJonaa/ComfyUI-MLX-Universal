## Doc-Code Discrepancies Found

- **mlx-embeddings missing from docs (Doc lagging):** The codebase implements `MLXTextEmbedding` in `nodes/embedding_nodes.py`, but it was missing from the Core Capabilities table and Architecture Map.
- **Kokoro TTS missing from docs (Doc lagging):** The codebase implements `MLXKokoroTTS` in `nodes/audio_nodes.py`, but the Audio capability in the README only mentioned `mlx-whisper`.
- **mlx-lm text generation crash due to thinking tokens (Structural debt):** `execute_text_generation` in `runtime/generate_processing.py` incorrectly passes vision-language specific arguments (`enable_thinking`, `thinking_budget`) to `mlx_lm.generate()`, which does not support them.
- **Missing cleanup of temp_img_path (Structural gap):** `execute_video_generation` in `runtime/video_processing.py` creates a temporary image file but fails to guarantee its deletion in the `finally` block.

## Documentation Changes

- **README.md**: Added `mlx-embeddings` as a new row in the Core Capabilities table.
- **README.md**: Appended `kokoro` to the Audio row in the Core Capabilities table.
- **README.md**: Updated the Architecture Map mermaid chart to correctly model the frontend to runtime routing for `MLXKokoroTTS` and `MLXTextEmbedding`.

## Roadmap Changes

Updated header `Last curated:` to `2026-07-12 at commit fbb4f37`.

Added: `[RM-018] Fix mlx-lm text generation crash due to thinking tokens`
- Evidence: `execute_text_generation` in `runtime/generate_processing.py` passing unsupported kwargs to `mlx_lm.generate()`.
- Why it matters: Prevents crashes during standard text generation.

Added: `[RM-020] Guarantee cleanup of temporary image paths in video generation`
- Evidence: `temp_img_path` omitted from `os.remove` in the `finally` block of `runtime/video_processing.py`.
- Why it matters: Prevents silent disk leaks after repeated generations or failures.

## Code Annotations Added

`runtime/diffusion_processing.py:126` — Added explanation for `mx.eval(latents)` in `encode_image` to explain that it prevents uncomputed lazy arrays from entering the bridge layer, which causes ComfyUI deadlocks.

## Open Questions

None this cycle.

## Verification

- Verified zero logic changes and zero parameter key changes across all modified files.
- Verified roadmap header timestamp and commit SHA point to current HEAD.
