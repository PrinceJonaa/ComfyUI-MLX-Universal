# The local tests failed with `load_flux_pipeline` and `track_audio_model` missing on REAL_MLX run?
# Wait! In the mock configuration inside `tests/test_helper.py`:
# When `REAL_MLX_TESTS=1`, I DO NOT mock `torch`, I DO NOT mock `comfyui_mlx_universal.runtime.*`.
# So `load_flux_pipeline` should be in `comfyui_mlx_universal.runtime.model_loader`.
# Why did it raise `AttributeError` when I ran `REAL_MLX_TESTS=1` locally?
# Look closely at `test_helper.py`:
# ```python
# def import_submodule(subfolder, name):
#     try:
#         spec.loader.exec_module(module)
#     except Exception as e:
#         print(f"Warning: could not execute {full_name}: {e}")
#     return module
# ```
# If `libmlx.so` is missing, `registry.py` throws ImportError.
# This causes `exec_module(module)` to FAIL.
# It prints the warning: "Warning: could not execute comfyui_mlx_universal.runtime.model_loader: cannot import name 'get_or_load_model' from 'comfyui_mlx_universal.runtime.registry'".
# BUT it returns an EMPTY `module` object.
# The EMPTY `module` object is put into `sys.modules["comfyui_mlx_universal.runtime.model_loader"]`.
# Since it's empty, it has no attributes!
# So `AttributeError` happens later.
# This was ONLY because of MY LOCAL `libmlx.so` missing.
# IN APPLE SILICON CI, `libmlx.so` IS PRESENT.
# Therefore, `exec_module` SUCCEEDS.
# Therefore, `sys.modules["comfyui_mlx_universal.runtime.model_loader"]` is FULL.
# So `AttributeError` DOES NOT HAPPEN IN CI.

# So why did the Apple Silicon CI fail?
# "File: .github, Line: 96, Message: Process completed with exit code 1."
# Could it be because `test_generate_nodes` failed?
# Let's check why `test_generate_nodes` might fail in CI.
# Does `test_generate_nodes` use `image="string"` or something else that fails MyPy or breaks?
