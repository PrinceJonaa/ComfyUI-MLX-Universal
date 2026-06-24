## Doc-Code Discrepancies Found

- The diagram in `README.md` claimed display names `MLX Generate Image`, `MLX Transcribe Audio`, and `MLX VAE Decode` but these lag behind code which are actually `MLX Generate Image (Flux)`, `MLX Transcribe Audio (Whisper)`, and `MLX VAE Decode (Flux)`. (Doc lagging)
- The roadmap `[RM-011] Refactor audio dimension reduction` was marked as "Planned" while the codebase already correctly uses `waveform[0]` instead of `.squeeze(0)`. (Doc lagging)

## Documentation Changes

- Synced `README.md`'s Architecture Map by updating the orphaned UI nodes into the graph to use the exact names from `NODE_DISPLAY_NAME_MAPPINGS`.

## Roadmap Changes

Updated: [RM-011] Refactor audio dimension reduction -> Completed (Evidence: `nodes/audio_nodes.py` now uses `waveform[0]` instead of `.squeeze(0)` for batch dimension reduction.)
Updated: The `Last curated` stamp in `roadmap.md` has been updated to the current date and commit HEAD `4b089e6`.

## Code Annotations Added

- `nodes/audio_nodes.py:27` — `# Prevents ComfyUI from crashing on startup in unsupported environments` — added to explain the lazy import of `mlx_whisper`.
- `nodes/system_nodes.py:11` — `# Prevents the workflow execution engine from aggressively caching nodes with no required inputs` — added to explain why `IS_CHANGED` returns `NaN`.
- `nodes/system_nodes.py:35` — `# Prevents the workflow execution engine from aggressively caching nodes with no required inputs` — added to explain why `IS_CHANGED` returns `NaN`.
- `runtime/bridge.py:53` — `# Avoids conversion errors with tensors requiring gradients or residing on a GPU` — added to explain the `.cpu().detach().numpy()` conversion.

## Open Questions

None.

## Verification

Confirmed zero logic changes and zero parameter key changes. `roadmap.md` header stamp updated to current HEAD commit. Test suite ran and passed verifying the modules import and run successfully.