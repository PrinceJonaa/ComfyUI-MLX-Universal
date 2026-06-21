# Roadmap — ComfyUI-MLX-Universal

> Last curated: 2026-06-20 at commit e157412
> This file reflects verified current state, not aspiration. Every entry has
> supporting evidence in the codebase or commit history — no entry is here
> on a guess.

## In Progress

## Planned

### [RM-001] Add Speculative Decoding and Thinking Tokens to LLM Node
- Status: Planned
- Evidence: `README.md` claims `mlx-lm` has "speculative decoding, thinking tokens", but `nodes/generate_nodes.py`'s `MLXLMGenerateText` lacks parameters for them, whereas `MLXVLMDescribeImage` has them.
- Why it matters: Keeps text generation capabilities on par with visual capabilities and fulfills the advertised feature set.

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

## Blocked

## Recently Completed

### [RM-002] Implement IS_CHANGED for System Nodes — completed 2026-06-20
- Evidence: `nodes/system_nodes.py` now implements `IS_CHANGED` for `MLXClearCache` and `MLXCacheStats`.

## Deferred / Rejected
- **[RM-006] Registry Tracking and Tensor Bridge Conversions** — removed 2026-06-20. Reason: already done in prior cycle.
