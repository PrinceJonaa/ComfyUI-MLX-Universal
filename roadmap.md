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

### [RM-002] Implement IS_CHANGED for System Nodes
- Status: Planned
- Evidence: `nodes/system_nodes.py` nodes (`MLXClearCache`, `MLXCacheStats`) lack required inputs but missing `IS_CHANGED` method.
- Why it matters: Without `IS_CHANGED`, ComfyUI will cache these nodes and only run them once per session, preventing caching stats and cache clearing from working iteratively.

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

### [RM-006] Registry Tracking and Tensor Bridge Conversions
- Status: Completed
- Evidence: `runtime/registry.py` and `runtime/bridge.py` are fully implemented, used across codebase.

## Deferred / Rejected
