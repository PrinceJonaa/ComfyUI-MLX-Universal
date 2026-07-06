## What This Fixes

This PR introduces the highly requested "Batch Understand Image" feature for Mac users running ComfyUI. Previously, describing multiple images required duplicating node structures or complex custom routing. Now, users can natively batch process lists of images against Vision-Language Models (VLMs) sequentially within a single workflow.

## What Changed

- **Runtime Substrate**: Added `execute_batch_image_description` to `runtime/generate_processing.py`. This explicitly loops over multiple PIL images generated from batched ComfyUI tensors and generates descriptions for each independently, avoiding massive parallel OOM crashes on Apple Silicon unified memory.
- **Node Interface**: Added `MLXVLMBatchDescribeImage` to `nodes/generate_nodes.py` which exposes the batch endpoint safely.
- **Batch Signalling**: Utilized `OUTPUT_IS_LIST = (True,)` on the UI node to signal ComfyUI to handle the sequential outputs dynamically.
- **Testing**: Added rigorous edge-case mocks in `tests/test_runtime_generate.py` and `tests/test_generate_nodes.py` that confirm loop constraints.

## Verification

- Tested sequentially batch processing an array using unified MLX models.
- All standard checks (`make verify`) passed locally including linters, xenon complexity analysis, and exhaustive unit tests.
