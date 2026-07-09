## What This Fixes
Currently, the ComfyUI-MLX-Universal pack includes an `MLXVLMDescribeImage` node for Vision-Language tasks, but it only accepts a single image. This leaves a logical gap for Mac users wanting to process batches of images (which ComfyUI natively supports via batched `IMAGE` tensors).

## What Changed
- Added a new `execute_batch_image_description` runtime processing function that cleanly iterates over a batched `IMAGE` tensor and manages the `mlx_vlm` inference loop.
- Added a new ComfyUI Node `MLXBatchVLMDescribe` mapped to "MLX Batch Understand Image". It accepts the exact same parameters as the single-image node but utilizes `OUTPUT_IS_LIST = (True,)` so that the UI correctly treats the returned tuple as an iterable list of strings rather than a single monolithic string block.
- Implemented robust unit tests for both the runtime execution logic and the new node definition to prevent regressions.

This significantly enhances the utility of the MLX VLM nodes in complex workflows, allowing users to efficiently map multiple generated or loaded images through visual captioning or reasoning models in a single execution step.
