## Task Selected

Source: Roadmap RM-014
Why this one: This is a clear, contained task with a precise scope that reduces cache pressure and overhead for users by taking advantage of dynamic LoRA patching via `mlx_lm.utils.load_adapters`. It's a high-impact optimization for the core loading pipeline.

## Dedup Verification

gh available: no
Checked: branches (`git branch -r`), commit log (`git log --all --grep="RM-014"`), and ground-truth code read (`cat nodes/loader_nodes.py`).
Result: Confirmed unclaimed and unbuilt before starting.

## What Changed

- Modified `MLXApplyLoRA.apply_lora` in `nodes/loader_nodes.py` to check for `mlx_model.family == "mlx-lm"`.
- If true, dynamically patches the LoRA by deep-copying the internal model and calling `mlx_lm.utils.load_adapters` on it instead of rebuilding the unified model from the cache wrapper.
- Falls back to the previous load-time fusion logic for non-`mlx-lm` families (like `mlx-vlm`).
- Mocked `mlx_lm.utils` in `tests/test_helper.py` to prevent CI failures.
- Renamed and split the unit tests in `tests/test_loader_nodes.py` to assert both the new dynamic fusion and the fallback fusion mechanisms.

## Source Reconciliation

RM-014 status updated to Recently Completed in `roadmap.md` using the automated `roadmap.py` script.

## Skipped This Run

- `[RM-015]` and `[RM-016]` skipped because their scope (refactoring monolithic state dictionary mappings in `diffusionkit`) presents a much higher blast radius and risk of regression.
- `[RM-017]` skipped because `RM-014` acts as a direct optimization blocker for future adapter changes.

## Open Thread Responses

None pending.

## Verification

- `make test` executed and verified to pass completely.
- Confirmed zero parameter key renames or node schema changes in `nodes/loader_nodes.py`.
