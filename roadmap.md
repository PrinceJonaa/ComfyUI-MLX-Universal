# Roadmap — ComfyUI-MLX-Universal

> Last curated: 2026-06-22 at commit 3f23305
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

## Blocked

## Recently Completed

### [RM-001] Add Speculative Decoding and Thinking Tokens to LLM Node
- Status: Completed
- Evidence: Added `draft_model_path`, `enable_thinking`, and `thinking_budget` to `MLXLMGenerateText` node schema in `nodes/generate_nodes.py`, routing draft model requests to `mlx_lm.generate()`.

### [RM-003] Integrate Whisper/Kokoro via `mlx-audio` — completed 2026-06-22
- Evidence: `nodes/audio_nodes.py` now implements `MLXWhisperTranscribe` node utilizing `mlx-whisper`.

### [RM-002] Implement IS_CHANGED for System Nodes — completed 2026-06-20
- Evidence: `nodes/system_nodes.py` now implements `IS_CHANGED` for `MLXClearCache` and `MLXCacheStats`.

## Deferred / Rejected
- **[RM-006] Registry Tracking and Tensor Bridge Conversions** — removed 2026-06-20. Reason: already done in prior cycle.
