## Architectural Thesis

The `MLXSAM3Predictor.predict` method in `nodes/sam_nodes.py` directly instantiated the `Sam3Predictor` from `mlx_vlm` and invoked tensor-to-PIL conversion. This violated the strict separation invariant that UI nodes must remain lightweight wrappers and not directly interface with MLX execution logic. This PR extracts this logic into `execute_sam3_prediction` within the `runtime` substrate, preventing accidental memory exhaustion or dependency conflicts on node initialization and improving testing fault isolation.

## Debt Location

- `nodes/sam_nodes.py` lines 42-53 inside `MLXSAM3Predictor.predict`

## What Changed

- Created `execute_sam3_prediction` in `runtime/sam_processing.py` to handle PIL conversion, predictor initialization, and inference.
- Replaced the direct logic in `nodes/sam_nodes.py` with a simple call to the new `execute_sam3_prediction`.
- Migrated tests in `tests/test_sam_nodes.py` to mock the newly exported function instead of internal methods.
- Added a new unit test for `execute_sam3_prediction` in `tests/test_runtime_sam.py`.

## What Was Not Changed

- The `INPUT_TYPES`, `RETURN_TYPES`, and internal signature of `MLXSAM3Predictor` were entirely preserved. Existing ComfyUI workflows will load and deserialize seamlessly.
- The `process_sam3_result` logic remained functionally identical, only the invocation location shifted.

## Backward Compatibility

- Successfully executed the complete mock test suite via `make test` capturing UI logic flow, ensuring zero parameter regressions.
- Verified all node interfaces visually align with the standard schema via tests.

## Rejected Alternatives

- Leaving the inference logic inline and abstracting just the tensor conversions was rejected because `Sam3Predictor` instantiation still represented heavy MLX logic explicitly forbidden in the `nodes/` domain.

## Follow-On Candidates

- Investigate unifying SAM models via the global model cache `get_or_load_model` instead of reinstantiating `Sam3Predictor` per inference if memory pressure arises during sequential batch operations.
