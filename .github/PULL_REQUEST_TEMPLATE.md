## Context and Motivation
<!-- Why is this change needed? What problem does it solve? -->
<!-- If it resolves an open issue, please link it here (e.g., "Fixes #123") -->

## Summary of Changes
<!-- Provide a concise overview of what changed. -->
- 
- 

## Model & Performance Impact (MLX Specific)
<!-- If this PR introduces a new model modality (Audio, VAE, etc.) or alters an existing one, please detail the impact: -->
- **Memory Footprint:** Did you verify this model clears successfully from Unified Memory without leaking? 
- **Tensor Format:** Does this output standard ComfyUI tensors `[B, H, W, C]` scaled correctly `[0, 1]`?
- **Lazy Evaluation:** Did you explicitly call `mx.eval()` or force Torch conversion before yielding back to ComfyUI?

## Substrate Checklist for Reviewers
<!-- Please review the `CONTRIBUTING.md` guidelines before submitting. -->
- [ ] **No UI Business Logic:** I did not put heavy generation logic directly into the `nodes/` frontend.
- [ ] **Registry Cache:** I used `runtime/registry.py` (e.g., `get_or_load_model`) to load model weights.
- [ ] **Tensor Bridge:** I routed all raw MLX Arrays through `runtime/bridge.py` before returning them to ComfyUI.
- [ ] **Data Contracts:** I used the correct `@dataclass` structures (e.g., `LoadedMLXModel`) instead of unstructured dictionaries.
- [ ] **Style & Lint:** The code follows standard Python formatting and does not break existing workflows.

## Validation & Testing
<!-- How did you test this? Please provide a screenshot of the ComfyUI workflow if applicable. -->
