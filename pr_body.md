## Doc-Code Discrepancies Found

- **Core Capabilities Table**: Lagging. Added Kokoro TTS and Text Embeddings.
- **Architecture Map**: Lagging. Added `MLXDraftModelLoader`, `MLXTextEmbedding`, `MLXKokoroTTS`, and `MLXBatchVLMDescribeImage`.
- **Roadmap inline**: Ahead. Aspirational tasks should solely live in `roadmap.md`. Removed the inline version.
- **CONTRIBUTING.md mapping claim**: Lagging. The root `__init__.py` uses static dictionaries now, not dynamic imports. Updated.

## Documentation Changes

- Updated `README.md` to reflect verified `mlx-embeddings` and `kokoro` integration capabilities.
- Redrew the Mermaid diagram to account for the actual nodes present in `nodes/`.
- Cleared the redundant Phase 2 aspirational tracking from `README.md`.
- Corrected the new node registration instructions in `CONTRIBUTING.md`.

## Roadmap Changes

Added:
- [RM-019] Add `mlx-audio` to `requirements.txt` (Evidence: Used in `model_loader.py` but missing from requirements).
- [RM-018] Fix no-op `mx.eval()` call in `execute_batch_image_description` (Evidence: `runtime/generate_processing.py:196`).

Updated:
- Header stamp updated to `9affdf7` on `2024-10-24`.

## Code Annotations Added

- `runtime/bridge.py:41` - Explicit evaluation required because type casting creates a new uncomputed lazy array.
- `runtime/bridge.py:72` - Explicit evaluation required because type casting creates a new uncomputed lazy array.

## Open Questions

None.

## Verification

- Verified zero logic changes and zero parameter key changes across the diff.
- Confirmed the `roadmap.md` header stamp was updated to the current HEAD commit.
- Mypy, Ruff, and 45 tests successfully passed against the bridged abstractions.
