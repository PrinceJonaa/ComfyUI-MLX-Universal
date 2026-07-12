## What This Fixes
This feature addresses a logical gap where users could not efficiently process batches of images (such as video frames or image datasets) through Vision-Language Models in a single workflow execution. Previously, the VLM node only expected a single image.

## What Changed
- Added `execute_batch_image_description` to `runtime/generate_processing.py`. This iteratively evaluates a batched ComfyUI `IMAGE` tensor by extracting individual PIL images and applying `apply_chat_template` per element.
- It leverages strict `mx.eval()` and `mx.metal.clear_cache()` sweeps at the tail of each loop to guarantee that MLX computation graphs do not stack dynamically on Apple Silicon, preventing Out-Of-Memory (OOM) crashes on large batches.
- Created `MLXVLMBatchDescribeImages` in `nodes/generate_nodes.py`, duplicating the inputs of the standard VLM node but adding `OUTPUT_IS_LIST = (True,)`. This allows the output `[STRING]` list to fan out correctly to iterative ComfyUI downstream logic.
- Included full mock test coverage validating the new runtime batch loop in `tests/test_runtime_generate.py`.
