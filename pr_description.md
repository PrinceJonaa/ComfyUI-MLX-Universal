## Architectural Thesis

The MLX state dictionary weight mapping logic in `diffusionkit/mlx/model_io.py` suffered from severe "Duplicated boilerplate" debt. It utilized dozens of sequential dictionary comprehensions to perform string replacements on state dict keys. This was extremely inefficient, forcing the creation and destruction of a full new dictionary for every single replacement rule (causing memory thrashing) and spiking the McCabe complexity of `flux_state_dict_adjustments` to an F grade. This refactor specifically retires that debt by consolidating the chained comprehensions into a single-pass `_apply_key_replacements` abstraction.

## Debt Location

The technical debt was located in `diffusionkit/mlx/model_io.py` specifically within:
- `flux_state_dict_adjustments`
- `mmdit_state_dict_adjustments`
- `_common_vae_adjustments`

## What Changed

- Introduced a new helper function `_apply_key_replacements` that sequentially applies a list of string replacement tuples to keys in a single pass.
- Migrated all chained dictionary comprehensions in the targeted functions to use the new abstraction.
- Replaced the side-effect-heavy list comprehension anti-pattern `[state_dict.pop(k) for k in keys_to_pop]` with standard `for` loops to eliminate unnecessary memory allocation.
- McCabe complexity was significantly reduced (`flux_state_dict_adjustments` dropped from an F to a D; others dropped from D to C).

## What Was Not Changed

- No public node interfaces or schemas were modified.
- No new cross-dependencies between `nodes/` and `runtime/` were added.
- The literal output structure and tensor definitions of the generated `state_dict` dictionaries remain exactly identical; only the construction loop mechanism was changed.
- No changes to user-facing workflow serialization formats occurred.

## Backward Compatibility

- Successfully passed `make test` checking global application functionality and Node initialization/schema integrity.
- Confirmed backward compatibility by ensuring no `__init__` or node schema files were modified; ComfyUI node registrations are untouched.

## Rejected Alternatives

The alternative was to port the dictionary mappings to use Python 3.9's `str.removeprefix()` or more complex Regex engines. This was rejected because the existing `.replace` calls often targeted substrings embedded deeply in the keys (not just prefixes), and Regex overhead would introduce a different kind of performance debt without simplifying the code structure.

## Follow-On Candidates

- The `_apply_key_replacements` abstraction could potentially be expanded further to handle tensor splitting logic that still exists ad-hoc inside `flux_state_dict_adjustments`. This was deferred to maintain strict scope and prevent introducing regression risk in the tensor splitting math.
