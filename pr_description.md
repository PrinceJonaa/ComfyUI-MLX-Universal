## What This Fixes
Currently, the node pack allows users to describe a single image at a time via `MLXVLMDescribeImage`. This is a limitation for standard ComfyUI workflows that rely heavily on batch processing (e.g. captioning directories of images to train LoRAs or auto-tagging video frames).

This PR implements an explicit batch processing path for VLM image understanding.

## What Changed
*   **Runtime:** Added `execute_batch_image_description` in `runtime/generate_processing.py`. This function converts a batched ComfyUI `IMAGE` tensor into a list of PIL images and iteratively queries the VLM via `mlx_vlm.generate`.
*   **Nodes:** Added `MLXBatchVLMDescribeImage` in `nodes/generate_nodes.py`. This node acts as a thin wrapper for the batching logic, enforcing model type validation.
*   **Outputs:** The new node explicitly returns `OUTPUT_IS_LIST = (True, False)`, allowing the node to pass a list of strings (`text_list`) for downstream iterative nodes, while also returning a standard combined string (`concatenated_text`) for immediate viewing in text boxes.

## Tests
*   Added comprehensive unit tests simulating batched evaluation and error handling to `test_generate_nodes.py` and `test_runtime_generate.py`.

## Verification
*   Ran `make verify` enforcing complexity limits, linting, formatting, and unit tests (Passed).
