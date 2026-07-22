# 🚀 [Feature] Batch Text Generation Pipeline

## Description

**Objective:** Implemented missing `MLXBatchLMGenerateText` to complement `MLXBatchVLMDescribeImage`, fulfilling the logical gap for Batch Processing feature expansion requested by users, while safely maintaining the MLX/UI architecture.

## Details:
- **`nodes/batch_nodes.py`:** Added `MLXBatchLMGenerateText` node that dynamically processes a batch of multiple strings separated by double newlines (`\n\n`), returning output separated similarly.
- **`runtime/generate_processing.py`:** Engineered `execute_batch_text_generation` backend implementation running the true generation logic decoupled from the UI.
- Safety measures correctly implemented including explicit invocation of `mx.eval()` and `mx.metal.clear_cache()` inside the iteration loop to enforce Metal cache evaluation constraints without deadlocking memory on Macs.
