## What Was Broken
- **Silent Failures:** `video_processing.py` incorrectly used `process.poll()` to check the exit code before the process had fully terminated, potentially raising false positive errors or masking actual errors.
- **Architectural Violation:** `nodes/audio_nodes.py` contained heavy MLX instantiation logic, violating the Strict Substrate Rule which mandates that heavy operations must be delegated to `runtime/`.
- **Cache Bug:** `load_kokoro_pipeline` hardcoded the American language code (`lang_code="a"`), causing incorrect processing and caching when British voices were requested.
- **Fragile Tokenizer Fallback:** The fallback logic in `embedding_processing.py` only caught `ValueError`, causing `TypeError` crashes when certain tokenizers rejected `return_tensors="mlx"`.
- **Inaccurate Type Hints:** ComfyUI `IMAGE` outputs were incorrectly type-hinted as `dict` instead of `torch.Tensor` in video and SAM nodes.

## How I Discovered It
- Scanned the codebase for non-compliant logic (`import` statements inside node definitions without delegation to `runtime/`).
- Verified subprocess polling structures and detected the race condition in `video_processing.py`.
- Checked tokenizer fallbacks following the `knowledgebase_lookup` hint and expanded exception handling.

## Verification
- Extracted logic to `runtime/audio_processing.py` successfully and verified that the nodes function identically as light wrappers.
- Executed `make verify` and corrected subsequent mock test failures by updating `tests/test_helper.py` and `patch_torch_imports`. All 37 local tests pass and the complexity score remains strong.

## Issues Resolved
- Improves general reliability, memory safety (via proper caching paths), and test suite accuracy.
