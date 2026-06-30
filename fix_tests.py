import re

# Since `encode` now requires `torch.Tensor` type hint, passing a string `"image_tensor"` in tests will fail MyPy if MyPy ran on tests, but we don't run MyPy on tests.
# Wait, why did the test fail?
# The output of `REAL_MLX_TESTS=1 python3 -m unittest discover tests -v` locally was:
# ERROR: test_mlx_load_flux_happy_path (test_diffusion_nodes.TestDiffusionNodes.test_mlx_load_flux_happy_path)
# AttributeError: <module 'comfyui_mlx_universal.runtime.model_loader' from '/app/runtime/model_loader.py'> does not have the attribute 'load_flux_pipeline'

# BUT `test_mlx_encoder_happy_path` PASSED!!
# Look at my log:
# test_mlx_encoder_happy_path (test_diffusion_nodes.TestDiffusionNodes.test_mlx_encoder_happy_path) ... ok
# test_mlx_decoder_happy_path ... ok

# Ah! The failure I saw `test_generate_nodes` and `test_loader_nodes` is because of MISSING IMPORTS or removed methods!
# Why did Apple Silicon CI fail?
# Let's read the Github Actions again:
# "Process completed with exit code 1."
# The only reason Apple Silicon CI fails is if the unit tests fail.
# If they fail, and it's because of `test_generate_nodes` or `test_runtime_audio` or something else that also failed locally...
# Wait, why do they fail locally?
# `AttributeError: <module ... model_loader> does not have the attribute 'track_audio_model'`
# `AttributeError: <module ... model_loader> does not have the attribute 'load_flux_pipeline'`
# Look at `runtime/model_loader.py`:
