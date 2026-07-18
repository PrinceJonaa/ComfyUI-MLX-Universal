## Architectural Thesis

The `flux_state_dict_adjustments`, `mmdit_state_dict_adjustments`, and `_common_vae_adjustments` functions in `diffusionkit/mlx/model_io.py` were massive monolithic blocks of sequential string replacements and complex splitting logic. This high cyclomatic complexity (flagged with 'F' and 'D' grades by `radon`) made it difficult to extend, test, and maintain these mappings. By modularizing these functions with declarative replacement dictionaries and abstracted helper functions for tensor splitting, we drastically reduce the cyclomatic complexity (now C and B grades) while maintaining exact functional behavior.

## Debt Location

- `diffusionkit/mlx/model_io.py`: Lines 99-198 (`flux_state_dict_adjustments`), Lines 200-241 (`mmdit_state_dict_adjustments`), and Lines 243-281 (`_common_vae_adjustments`).

## What Changed

- Abstracted sequential string replacement logic into a general `_apply_replacements` helper function that processes replacements using declarative mapping dictionaries.
- Extracted complex tensor splitting logic from `flux_state_dict_adjustments` into distinct helper methods (`_split_flux_qkv`, `_split_flux_linear1`, `_split_flux_linear2`).
- Replaced the large conditional `if/elif` blocks in `_common_vae_adjustments` with a linear flow matching the original sequential execution order while still utilizing dictionary replacements.

## What Was Not Changed

- All MLX model architectures, inference loops, and user-facing CLI inputs remain identical.
- No public interfaces, node schemas, or serialization formats were altered. This refactor is purely internal to the state dict manipulation pipeline.
- Other functions inside `diffusionkit/mlx/model_io.py` not related to the stated monolithic adjustments (e.g. text encoders) were kept as is.

## Backward Compatibility

- Executed `make test` locally to ensure that all models still load properly and unit tests pass without errors.
- The `make complexity` validation was performed using `radon` and `xenon` to guarantee that cyclomatic complexity metrics have tangibly improved from 'F' to 'C' or better for the refactored logic.

## Rejected Alternatives

- **Refactoring dynamic LoRA fusion (RM-014)**: Was rejected during this cycle because MLX adapter APIs couldn't be definitively tested/verified without physical Apple Silicon access, risking untested runtime OOM crashes or immutability violations in the ComfyUI DAG cache.

## Follow-On Candidates

- The `t5_encoder_state_dict_adjustments` and various `map_vae_weights` helper functions could also be transitioned to the new declarative `_apply_replacements` pattern for consistency, but were deferred to limit the scope of this structural refactor to the most complex code paths.
