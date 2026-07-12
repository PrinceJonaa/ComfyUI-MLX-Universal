## Task Selected

Source: Roadmap RM-014
Why this one: This is a clearly defined, single-task roadmap item with a contained blast radius that prevents UI nodes from redundantly reloading entire base models just to fuse a LoRA adapter.

## Dedup Verification

gh available: no
Checked: branches, commit log, ground-truth code read (roadmap RM-014 was Planned and `nodes/loader_nodes.py` was indeed reloading).
Result: confirmed unclaimed and unbuilt before starting

## What Changed

Refactored `nodes/loader_nodes.py` so `MLXApplyLoRA` no longer calls `load_unified_mlx_model`.
Introduced `apply_dynamic_lora` in `runtime/model_loader.py`. This uses `copy.deepcopy()` to clone the MLX model graph in-memory, then delegates to `mlx_lm`'s `load_adapters` or `mlx_vlm`'s `apply_lora_layers` to fuse weights, eliminating redundant I/O while protecting the base model in the unified cache pool.
Updated `tests/test_loader_nodes.py` to mock and verify this dynamic fusion path.

## Source Reconciliation

RM-014 status updated to Recently Completed via `scripts/roadmap.py`.

## Skipped This Run

None (first candidate from roadmap selected).

## Open Thread Responses

none pending.

## Verification

make test passed successfully after updating tests for the new runtime mock. No schema/param keys changed in `nodes/loader_nodes.py`.
