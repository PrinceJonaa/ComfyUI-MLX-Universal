# Ah! The ACTUAL CI log from GitHub Actions is at the TOP of the prompt:
#
# ERROR: test_generate_nodes (unittest.loader._FailedTest.test_generate_nodes)
# ImportError: Failed to import test module: test_generate_nodes
# Traceback (most recent call last):
#   File "/Users/runner/work/ComfyUI-MLX-Universal/ComfyUI-MLX-Universal/tests/test_generate_nodes.py", line 8, in <module>
#     mock_mlx_lm = sys.modules["mlx_lm"]
# KeyError: 'mlx_lm'
#
# ERROR: test_encode_clip_text (test_runtime_diffusion.TestRuntimeDiffusion.test_encode_clip_text)
# Traceback:
#   File "/Users/runner/work/ComfyUI-MLX-Universal/ComfyUI-MLX-Universal/runtime/diffusion_processing.py", line 96, in _tokenize
#     return mx.array(tokens)
# ValueError: Invalid type unittest.mock.MagicMock received in array initialization.
#
# ERROR: test_predict_happy_path (test_sam_nodes.TestSAMNodes.test_predict_happy_path)
# Traceback:
#   File "/Users/runner/work/ComfyUI-MLX-Universal/ComfyUI-MLX-Universal/tests/test_sam_nodes.py", line 36, in test_predict_happy_path
#     mock_sam_predictor_cls = sys.modules["mlx_vlm.models.sam3.generate"].Sam3Predictor
# KeyError: 'mlx_vlm.models.sam3.generate'
#
# FAIL: test_mlx_to_torch_and_back (test_runtime_bridge.TestRuntimeBridge.test_mlx_to_torch_and_back)
# Traceback:
#   File "/Users/runner/work/ComfyUI-MLX-Universal/ComfyUI-MLX-Universal/tests/test_runtime_bridge.py", line 25, in test_mlx_to_torch_and_back
#     self.assertIsInstance(torch_tensor, torch.Tensor)
# AssertionError: <MagicMock name='mock()' id='4516587792'> is not an instance of <class 'torch.Tensor'>
#
# This means:
# In the REAL GitHub Actions CI (on Mac), `USE_REAL_MLX` is set to `1` (which skips mocking `mlx_lm`, `mlx_vlm`, etc. in `test_helper.py`).
# BUT the test files (`test_generate_nodes.py`, `test_sam_nodes.py`, etc.) are unconditionally doing:
# `mock_mlx_lm = sys.modules["mlx_lm"]`
# Which fails with `KeyError` because it hasn't been imported yet, or it's not mocked!
