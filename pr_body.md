## Tasks Completed

- **Type Errors**: Fixed `[attr-defined]` errors in `tests/test_helper.py` by swapping direct module attribute assignment for `setattr()`. This resolves errors when mocking out modules.
- **Type Errors**: Fixed `[union-attr]` error in `runtime/generate_processing.py` related to `tokenizer` methods. Added a `None` check and used `# type: ignore[union-attr]` since `mypy` cannot perfectly narrow types when relying on `hasattr` or `getattr` on an `Any | None` object.
- **.gitignore**: The git cache was cleaned to remove tracked files that should be ignored by the current `.gitignore` (specifically ensuring `.DS_Store`, `__pycache__`, `.env`, and `.venv` are untracked).
- **Formatting**: The codebase was formatted with `ruff`.

## Specific Linters/Rules applied

- **Mypy**: Enforced strict typing for dynamic module mocking and attribute checking.
- **Ruff**: Applied standard python formatting and safe fixes.
- **Git**: Ensured the working tree adheres to the repository's `.gitignore` policy.
