## Doc-Code Discrepancies Found
- **Doc lagging code:** `README.md` listed Phase 1 as "(In Progress)" with an unchecked box for unified pipelines, but `nodes/` and `runtime/` now fully implement Text, Vision, Samplers, and Video pipelines. Fixed the doc.
- **Doc ahead of code / orphaned claim:** `roadmap.md` had an active task `[RM-010] Native Kokoro Integration` referencing a `README.md` claim. The README no longer claims Kokoro integration. Removed the task to Deferred/Rejected.
- **Roadmap lagging code:** `roadmap.md` had an active task `[RM-009] Enforce dict return type hints for INPUT_TYPES`. The trace confirmed all nodes in `nodes/` now correctly type hint `-> dict:`. Moved to Recently Completed.

## Documentation Changes
- Updated `README.md` Phase 1 from "(In Progress)" to "(Completed)" and checked the final box for unified pipelines to match the codebase reality.

## Roadmap Changes
- Updated: `roadmap.md` header stamp updated to `2026-06-26` at commit `e1ee695`.
- Updated: `[RM-009] Enforce dict return type hints for INPUT_TYPES` moved from Planned to Recently Completed.
- Deferred/Removed: `[RM-010] Native Kokoro Integration` moved to Deferred/Rejected because the README no longer claims this integration, making the gap obsolete.

## Code Annotations Added
- `nodes/diffusion_nodes.py:211` — `mx.eval(t5_embeddings, clip_pooled_output)` — Added comment explaining this forces simultaneous evaluation to avoid deferred computation slowing down the diffusion loop.
- `nodes/loader_nodes.py:92` — `apply_lora` — Added comment explaining that LoRA weights are fused at load-time to ensure safe tracking within the MLX unified memory cache.

## Open Questions
- None.

## Verification
- Verified zero logic changes and zero parameter key changes.
- Verified roadmap.md header stamp is updated to the current HEAD commit.
