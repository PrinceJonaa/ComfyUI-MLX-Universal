## Doc-Code Discrepancies Found

- The README.md was lagging the codebase in several areas:
  - Missing the Kokoro TTS implementation present in `nodes/audio_nodes.py` (added as `MLXKokoroTTS`).
  - Missing the Text Embedding implementation present in `nodes/embedding_nodes.py` (added as `MLXTextEmbedding`).
  - The architecture graph omitted `MLXBatchVLMDescribeImage` in `nodes/batch_nodes.py`, as well as `MLXTextEmbedding` and `MLXKokoroTTS`.

## Documentation Changes

- Updated `README.md` to include Kokoro and Embeddings in the Core Capabilities table.
- Updated `README.md`'s Mermaid Architecture Map to accurately reflect all backend `MLX` wrapper nodes (`UI_BatchVLM`, `UI_Embed`, `UI_TTS`) and their target files.

## Roadmap Changes

- Updated the `Last curated` stamp to the current execution date (2024-07-22) and HEAD commit SHA (9affdf7) in `roadmap.md` to prepare the diff context for the next cycle.

## Code Annotations Added

- `runtime/generate_processing.py:27` (and corresponding batch functions) — added `# explicit seeding at execution time prevents ComfyUI's global seed from being swallowed by MLX's internal PRNG state` to clarify the non-obvious requirement of the explicit `mx.random.seed(seed)` call prior to LLM/VLM generation inference.

## Open Questions

None. All mismatches were clear cases of "doc lagging code."

## Verification

- Confirm: zero logic changes.
- Confirm: zero parameter key changes.
- Confirm: `roadmap.md` header stamp updated to current HEAD commit.
