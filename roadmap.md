# Roadmap — ComfyUI-MLX-Universal

> Last curated: 2026-06-21 at commit e157412
> This file reflects verified current state, not aspiration. Every entry has
> supporting evidence in the codebase or commit history — no entry is here
> on a guess.

## In Progress

## Planned

### [RM-007] Enforce ComfyUI Latent Dictionary Format and Tensor Bridging in Diffusion Nodes
- Status: Planned
- Evidence: `nodes/diffusion_nodes.py`'s `MLXSampler` returns a raw MLX array `(latents,)` instead of the required ComfyUI format `{"samples": torch_tensor}` and skips `runtime/bridge.py`.
- Why it matters: Breaks downstream ComfyUI nodes expecting standard PyTorch latent format, violating the architecture tensor bridge rules.

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

### [RM-002] Implement IS_CHANGED for System Nodes — completed 2026-06-21
- Evidence: `nodes/system_nodes.py` already implements `IS_CHANGED` returning `float("NaN")`.

### [RM-006] Registry Tracking and Tensor Bridge Conversions
- Status: Completed
- Evidence: `runtime/registry.py` and `runtime/bridge.py` are fully implemented, used across codebase.

## Deferred / Rejected
- **[RM-001] Add Speculative Decoding and Thinking Tokens to LLM Node** — removed 2026-06-21. Reason: README claim is already accurate; speculative decoding/thinking tokens are correctly attributed to VLMs, not LLMs.
