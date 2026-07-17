[Fix] Audio and Embeddings: Fix Kokoro TTS language codes and tokenizer encode exception handling

## What was broken
- `KokoroPipeline` was hardcoded to `lang_code="a"`, meaning voices starting with `b` (British) were loaded with the wrong voice code.
- `tokenizer.encode` fallback in `runtime/embedding_processing.py` didn't catch `TypeError` thrown by some huggingface tokenizers when `return_tensors="mlx"` was passed.

## How it was discovered
- Reviewed `KokoroTTS` loading which indicated that a pipeline could use `a` or `b` depending on the voice model prefix.
- Memory directed that huggingface tokenizers could raise either `ValueError` or `TypeError` during unsupported tensor types.

## How the fix was verified
- Automated test suite passes successfully.
- Code changes were reviewed against the codebase context and explicitly checked using `ruff` and `make verify`.

## Issues resolved
- None explicitly, but fixes hidden edge-cases.

## Explicitly Out of Scope
- Dynamic LoRA fusion via `MLXApplyLoRA` (RM-014) was attempted but reverted because `model.update()` mutates the global cache in ComfyUI, breaking DAG immutability, and full `nn.Linear` to `LoRALinear` conversion would be too brittle without proper MLX tree map utilities. Load-time fusion remains the safest approach for now.
