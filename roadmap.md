# Roadmap — ComfyUI-MLX-Universal

> Last curated: 2026-06-22 at commit 3f23305
> This file reflects verified current state, not aspiration. Every entry has
> supporting evidence in the codebase or commit history — no entry is here
> on a guess.

## In Progress

## Planned

### [RM-003] Integrate Whisper/Kokoro via `mlx-audio`
- Status: Planned
- Evidence: `README.md` claims Audio in Phase 2 Expansion.
- Why it matters: Achieves the ultimate goal of being the definitive "One-Stop Shop" for all modalities.

### [RM-004] Native SDXL / ControlNet pipelines
- Status: Planned
- Evidence: `README.md` claims SDXL / ControlNet in Phase 2 Expansion.
- Why it matters: Achieves the ultimate goal of being the definitive "One-Stop Shop" for all modalities.

### [RM-005] Standalone Causal Video and Image VAE encode/decode nodes
- Status: Planned
- Evidence: `README.md` claims VAEs in Phase 2 Expansion.
- Why it matters: Achieves the ultimate goal of being the definitive "One-Stop Shop" for all modalities.

### [RM-007] Refactor SAM3 node to remove PIL/drawing logic
- Status: Planned
- Evidence: `nodes/sam_nodes.py` contains heavy array/PIL manipulation and drawing logic inside the UI node, violating the strict separation of concerns in `CONTRIBUTING.md`.
- Why it matters: Maintains architecture boundaries and ensures UI wrappers stay thin.

### [RM-008] Extract video generation subprocess and CV2 logic
- Status: Planned
- Evidence: `nodes/video_nodes.py` directly handles `subprocess.Popen`, temp file management, and `cv2` video reading inside the ComfyUI wrapper class.
- Why it matters: Moves heavy background logic out of UI files into the runtime substrate.

### [RM-009] Enforce dict return type hints for INPUT_TYPES
- Status: Planned
- Evidence: Multiple nodes in `nodes/` implement `INPUT_TYPES(s)` without a return type hint (e.g., `-> dict:`), causing static analysis drift.
- Why it matters: Improves strict code cleanliness and static analysis verification for the API.

## Blocked

## Recently Completed

### [RM-002] Implement IS_CHANGED for System Nodes — completed 2026-06-20
- Evidence: `nodes/system_nodes.py` now implements `IS_CHANGED` for `MLXClearCache` and `MLXCacheStats`.

## Deferred / Rejected
- **[RM-001] Add Speculative Decoding and Thinking Tokens to LLM Node** — removed 2026-06-22. Reason: False premise; README correctly scopes these features to VLMs, not LLMs.
- **[RM-006] Registry Tracking and Tensor Bridge Conversions** — removed 2026-06-20. Reason: already done in prior cycle.
