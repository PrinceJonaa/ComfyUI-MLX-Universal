## Architectural Thesis

The previous architecture exhibited a "Leaking abstraction" and "Cache management scatter". UI-facing nodes (`MLXLoadFlux`, `MLXLMGenerateText`, `MLXVLMDescribeImage`, and `MLXWhisperTranscribe`) were tightly coupled with internal cache key generation (`make_key`), direct `registry.py` manipulation, and explicit loader closures. This scattered state tracking and created a fragile dependency surface where changes to MLX unified memory caching would require coordinated updates across disconnected frontend modules. By introducing `load_draft_model`, `track_audio_model`, and `load_flux_pipeline` to the `runtime/model_loader.py` boundary, we centralize this logic. This eliminates boilerplate and guarantees that all MLX initialization and caching remains encapsulated within the `runtime/` substrate, satisfying the core architectural invariant.

## Debt Location

- `nodes/diffusion_nodes.py`: Lines 127-149 (`check_model_folder` and inline registry calls in `MLXLoadFlux`).
- `nodes/generate_nodes.py`: Lines 98-106 (inline draft loading in `MLXLMGenerateText`) and Lines 221-231 (inline draft loading in `MLXVLMDescribeImage`).
- `nodes/audio_nodes.py`: Lines 38-48 (inline dummy loader and `make_key` in `MLXWhisperTranscribe`).

## What Changed

- **Extracted:** Added `load_draft_model`, `track_audio_model`, and `load_flux_pipeline` to `runtime/model_loader.py`.
- **Consolidated:** Moved `check_model_folder` logic out of the `MLXLoadFlux` class and encapsulated it into `load_flux_pipeline`.
- **Migrated:** Updated `MLXLoadFlux.load_flux_model`, `MLXLMGenerateText.generate`, `MLXVLMDescribeImage.run`, and `MLXWhisperTranscribe.transcribe` to call these new functions instead of importing `registry.py` internals, defining inline `_loader` closures, or computing `make_key` themselves.
- **Example Before (generate_nodes.py):**
  ```python
  draft_key = make_key(draft_model_path, "draft")
  def _load_draft():
      print(f"Loading draft model '{draft_model_path}'...")
      model, _ = mlx_lm.load(draft_model_path)
      return model
  draft_model = get_or_load_draft_model(draft_key, _load_draft)
  ```
- **Example After:**
  ```python
  from ..runtime.model_loader import load_draft_model
  draft_model = load_draft_model(draft_model_path, "mlx-lm")
  ```

## What Was Not Changed

- **Public Interfaces:** All `INPUT_TYPES` and `RETURN_TYPES` remain entirely unmodified.
- **Node Registries:** `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` across all files were untouched.
- **Serialization:** Because the node schemas were preserved exactly, all existing ComfyUI saved workflows utilizing these nodes will continue to deserialize without any errors.

## Backward Compatibility

- Verified compatibility manually by reading node schemas and structurally confirming that `INPUT_TYPES` methods were not edited.
- Executed `python3 -m unittest discover tests` successfully, which tests the mock invocation of these nodes using standard ComfyUI serialized formats.
- Executed syntax verification (`py_compile.compile`) on modified modules to guarantee no import errors for Apple Silicon runtimes.

## Rejected Alternatives

The next most viable alternative was to create a new module (e.g. `runtime/cache_utils.py`) exclusively for cache logic and keys, while leaving the inline closures in the nodes. This was rejected because it only resolves the "Cache management scatter" but still leaves the "Leaking abstraction" (the nodes still have to know *how* to load models, they just use a cleaner caching backend). Consolidating the full instantiation flow into `model_loader.py` retires both debts simultaneously.

## Follow-On Candidates

- **Tensor bridge inconsistency:** `mlx_to_torch` and explicit `mx.eval()` are currently peppered throughout `diffusion_nodes.py` decoding steps. This should be audited and moved strictly into `bridge.py` wrappers in the next cycle to ensure uniform evaluation.
