## Architectural Thesis

The MLX diffusion implementation details (lazy evaluation triggers, tokenization logic, and complex MLX tensor manipulations) were heavily bleeding into the ComfyUI node wrappers in `nodes/diffusion_nodes.py`. This created a severely leaking abstraction where the UI layer had to juggle core dependencies (like `mlx.core` and `diffusionkit`) and perform low-level caching. By introducing `runtime/diffusion_processing.py` as a dedicated barrier, we isolate these responsibilities strictly to the runtime substrate, reducing dependency drift and completely satisfying the strict UI/MLX separation invariant.

## Debt Location

- `nodes/diffusion_nodes.py`: Execution functions inside `MLXDecoder`, `MLXSampler`, `MLXClipTextEncoder`, and `MLXEncoder` classes.
- `nodes/diffusion_nodes.py`: The `_tokenize` private helper method residing in the UI layer.

## What Changed

- Introduced `runtime/diffusion_processing.py` containing four high-level execution functions: `decode_latents`, `generate_image`, `encode_clip_text`, and `encode_image`.
- Migrated all `mx.eval()` calls, tokenization scaling logic, and diffusion tracking into these runtime functions.
- Stripped all `mlx.core` and `diffusionkit` imports out of `nodes/diffusion_nodes.py`, converting it into a pure dictionary-passing frontend wrapper.
- Extracted the `_tokenize` helper method out of the `MLXClipTextEncoder` class and into the runtime module.

## What Was Not Changed

- All ComfyUI node definitions (`INPUT_TYPES`, `RETURN_TYPES`, `RETURN_NAMES`) remain 100% identical.
- The public signatures and parameter keys are untouched. Existing saved user workflows will successfully deserialize and run without any modification.
- Existing internal bridges (e.g., `runtime/bridge.py` and `runtime/model_loader.py`) remain completely unchanged, ensuring cache behaviors are unaffected.

## Backward Compatibility

- Ran `python3 -m unittest discover tests` successfully across all 20 existing unit tests, explicitly verifying that the newly abstracted node methods still correctly route and process mock MLX tensors via the bridge.
- Confirmed `mypy .` and `ruff check --fix .` pass with zero failures, proving robust typing and separation.
- Verified manual resolution of the project's roadmap merge conflicts without altering file tracking expectations.

## Rejected Alternatives

- **Extracting tokenization to `bridge.py` instead of `diffusion_processing.py`**: This was rejected because `bridge.py` is explicitly designed for agnostic framework conversion (PyTorch ↔ MLX), whereas tokenization is specific to the diffusion process. Dumping it there would violate single-responsibility.

## Follow-On Candidates

- Consider extracting the heavy dictionary parsing out of the node wrappers (like unpacking `mlx_positive_conditioning`) and handling parameter validation directly inside `diffusion_processing.py` to make the UI wrappers even thinner.
