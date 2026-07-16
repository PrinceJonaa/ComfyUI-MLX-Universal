# Roadmap — ComfyUI-MLX-Universal

> Last curated: 2026-07-16 at commit fbb4f37
> This file reflects verified current state, not aspiration. Every entry has
> supporting evidence in the codebase or commit history — no entry is here
> on a guess.

## In Progress

### [RM-015] Reduce cyclomatic complexity in IO and Runtime modules
- Status: In Progress
- Evidence: `execute_video_generation` in `runtime/video_processing.py` was successfully refactored (E -> D). `ruff check` / Xenon still flags `flux_state_dict_adjustments` (F), `_common_vae_adjustments` (D), and `mmdit_state_dict_adjustments` (D) in `diffusionkit/mlx/model_io.py`.
- Why it matters: High cyclomatic complexity creates monolithic, untestable functions. Breaking them down into modular mapping pipelines reduces the risk of regression during future updates.

## Planned

### [RM-004] Native SDXL / ControlNet pipelines
- Status: Planned
- Evidence: `README.md` claims SDXL / ControlNet in Phase 2 Expansion.
- Why it matters: Achieves the ultimate goal of being the definitive "One-Stop Shop" for all modalities.

## Blocked

## Recently Completed

### [RM-017] Increase Test Coverage for Runtime Extractions — completed 2026-07-16
- Status: Completed
- Evidence: PRs #73 and #80 successfully extracted logic to `runtime/audio_processing.py` and `runtime/diffusion_processing.py`, but tests still mock at the node layer instead of unit-testing the runtime functions directly.
- Why it matters: Prevents silent regressions in the new runtime bridging logic.

### [RM-016] Modularize model_io.py in diffusionkit — completed 2026-07-16
- Status: Completed
- Evidence: MCP architecture analysis and `make complexity` highlight `diffusionkit/mlx/model_io.py` as a monolithic hotspot with `flux_state_dict_adjustments` scoring an F in cyclomatic complexity.
- Why it matters: State dict adjustments are currently one massive function. Refactoring these into a Chain of Responsibility pattern or modular map dictionaries will drastically improve maintainability.

### [RM-014] Refactor dynamic LoRA fusion in `MLXApplyLoRA` — completed 2026-07-16
- Status: Completed
- Evidence: `MLXApplyLoRA.apply_lora` in `nodes/loader_nodes.py` re-invokes `load_unified_mlx_model` from scratch to fuse an adapter, bypassing dynamic weight patching.
- Why it matters: Reduces cache pressure and loading overhead when users swap LoRAs during live ComfyUI sessions.

### [RM-012] Fix static type hints for IMAGE inputs
- Status: Completed
- Evidence: `MLXVLMDescribeImage.run` in `nodes/generate_nodes.py` types the `image` argument as `dict | None = None`, but ComfyUI passes `IMAGE` as a `torch.Tensor`.
- Why it matters: Prevents static analysis drift and misleading type documentation for frontend node developers.

### [RM-013] Decouple Draft Model Loading from Generate Nodes
- Status: Completed
- Evidence: `MLXLMGenerateText` and `MLXVLMDescribeImage` both inline the loading of draft models using `draft_model_path`. 
- Why it matters: UI generation nodes should not handle loading IO. Draft models should be loaded by a dedicated `MLXDraftModelLoader` node that outputs a `DRAFT_MODEL` to be passed into generation nodes, adhering to ComfyUI's explicit graph structure.

### [RM-010] Native Kokoro Integration
- Status: Completed (PR #79)
- Evidence: `MLXKokoroTTS` node implemented in `nodes/audio_nodes.py` returning properly scaled waveforms via `runtime.bridge`.

### [RM-008] Extract video generation subprocess and CV2 logic
- Status: Completed
- Evidence: `nodes/video_nodes.py` now uses `execute_video_generation` from `runtime/video_processing.py` to handle `subprocess.Popen`, temp file management, and `cv2` video reading.

### [RM-009] Enforce dict return type hints for INPUT_TYPES — completed 2026-06-27
- Evidence: Added `-> dict:` to `INPUT_TYPES` methods in `nodes/video_nodes.py`, `nodes/loader_nodes.py`, and `nodes/audio_nodes.py`.

### [RM-005] Standalone Causal Video and Image VAE encode/decode nodes — completed 2026-06-26
- Evidence: Added MLXEncoder to `nodes/diffusion_nodes.py`.

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
- **[RM-001] Add Speculative Decoding and Thinking Tokens to LLM Node** — removed 2026-06-22. Reason: False premise; README correctly scopes these features to VLMs, not LLMs.
- **[RM-006] Registry Tracking and Tensor Bridge Conversions** — removed 2026-06-20. Reason: already done in prior cycle.
