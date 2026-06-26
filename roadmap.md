# Roadmap — ComfyUI-MLX-Universal

> Last curated: 2026-06-26 at commit e1ee695
> This file reflects verified current state, not aspiration. Every entry has
> supporting evidence in the codebase or commit history — no entry is here
> on a guess.

## In Progress

## Planned

### [RM-004] Native SDXL / ControlNet pipelines
- Status: Planned
- Evidence: `README.md` claims SDXL / ControlNet in Phase 2 Expansion.
- Why it matters: Achieves the ultimate goal of being the definitive "One-Stop Shop" for all modalities.

### [RM-005] Standalone Causal Video and Image VAE encode/decode nodes
- Status: Planned
- Evidence: `README.md` claims VAEs in Phase 2 Expansion.
- Why it matters: Achieves the ultimate goal of being the definitive "One-Stop Shop" for all modalities.

### [RM-008] Extract video generation subprocess and CV2 logic
- Status: Planned
- Evidence: `nodes/video_nodes.py` directly handles `subprocess.Popen`, temp file management, and `cv2` video reading inside the ComfyUI wrapper class.
- Why it matters: Moves heavy background logic out of UI files into the runtime substrate.

## Blocked

## Recently Completed

### [RM-009] Enforce dict return type hints for INPUT_TYPES — completed 2026-06-26
- Evidence: All nodes in `nodes/` directory now correctly include the `-> dict:` hint in their `INPUT_TYPES(s)` methods (e.g. `def INPUT_TYPES(s) -> dict:`).

### [RM-011] Refactor audio dimension reduction — completed 2026-06-24
- Evidence: `nodes/audio_nodes.py` now uses `waveform[0]` instead of `.squeeze(0)` for batch dimension reduction.

### [RM-007] Refactor SAM3 node to remove PIL/drawing logic
- Status: Completed
- Evidence: `nodes/sam_nodes.py` now uses `process_sam3_result` from `runtime/sam_processing.py` to handle PIL and tensor operations.

### [RM-001] Add Speculative Decoding and Thinking Tokens to LLM Node
- Status: Completed
- Evidence: Added `draft_model_path`, `enable_thinking`, and `thinking_budget` to `MLXLMGenerateText` node schema in `nodes/generate_nodes.py`, routing draft model requests to `mlx_lm.generate()`.

### [RM-003] Integrate Whisper/Kokoro via `mlx-audio` — completed 2026-06-22
- Evidence: `nodes/audio_nodes.py` now implements `MLXWhisperTranscribe` node utilizing `mlx-whisper`.

### [RM-002] Implement IS_CHANGED for System Nodes — completed 2026-06-20
- Evidence: `nodes/system_nodes.py` now implements `IS_CHANGED` for `MLXClearCache` and `MLXCacheStats`.

## Deferred / Rejected
- **[RM-010] Native Kokoro Integration** — removed 2026-06-26. Reason: README.md no longer claims Kokoro integration, gap no longer exists.
- **[RM-001] Add Speculative Decoding and Thinking Tokens to LLM Node** — removed 2026-06-22. Reason: False premise; README correctly scopes these features to VLMs, not LLMs.
- **[RM-006] Registry Tracking and Tensor Bridge Conversions** — removed 2026-06-20. Reason: already done in prior cycle.
