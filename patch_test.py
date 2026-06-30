import sys

def patch_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    # The issue is that the github actions run with `REAL_MLX_TESTS=1`.
    # When `REAL_MLX_TESTS=1`, the mock for `mlx_lm` is NOT created in `tests/test_helper.py`.
    # BUT `tests/test_generate_nodes.py` directly tries to access `sys.modules["mlx_lm"]` at the module level.
    # We should move the `sys.modules["mlx_lm"]` assignment inside the `setUp` or test methods, OR just wrap it in a try-except, OR check if it exists before assigning.

    # In test_generate_nodes.py:
    content = content.replace('mock_mlx_lm = sys.modules["mlx_lm"]', 'mock_mlx_lm = sys.modules.get("mlx_lm", None)')
    content = content.replace('mock_mlx_lm_sample_utils = sys.modules["mlx_lm.sample_utils"]', 'mock_mlx_lm_sample_utils = sys.modules.get("mlx_lm.sample_utils", None)')
    content = content.replace('mock_mlx_vlm = sys.modules["mlx_vlm"]', 'mock_mlx_vlm = sys.modules.get("mlx_vlm", None)')
    content = content.replace('mock_mlx_vlm_prompt_utils = sys.modules["mlx_vlm.prompt_utils"]', 'mock_mlx_vlm_prompt_utils = sys.modules.get("mlx_vlm.prompt_utils", None)')
    content = content.replace('mock_mlx_vlm_speculative = sys.modules["mlx_vlm.speculative"]', 'mock_mlx_vlm_speculative = sys.modules.get("mlx_vlm.speculative", None)')
    content = content.replace('mock_mlx_vlm_speculative_drafters = sys.modules["mlx_vlm.speculative.drafters"]', 'mock_mlx_vlm_speculative_drafters = sys.modules.get("mlx_vlm.speculative.drafters", None)')

    with open(file_path, "w") as f:
        f.write(content)

    print("File patched successfully.")

patch_file("tests/test_generate_nodes.py")
