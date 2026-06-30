import sys

def patch_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    # We also need to mock or conditionally handle things inside setUp since they will be None if not found

    content = content.replace("mock_mlx_lm.generate.reset_mock()", "if mock_mlx_lm: mock_mlx_lm.generate.reset_mock()")
    content = content.replace("mock_mlx_lm_sample_utils.make_sampler.reset_mock()", "if mock_mlx_lm_sample_utils: mock_mlx_lm_sample_utils.make_sampler.reset_mock()")
    content = content.replace("mock_mlx_vlm.generate.reset_mock()", "if mock_mlx_vlm: mock_mlx_vlm.generate.reset_mock()")
    content = content.replace("mock_mlx_vlm_prompt_utils.apply_chat_template.reset_mock()", "if mock_mlx_vlm_prompt_utils: mock_mlx_vlm_prompt_utils.apply_chat_template.reset_mock()")
    content = content.replace("mock_mlx_vlm_speculative_drafters.load_drafter.reset_mock()", "if mock_mlx_vlm_speculative_drafters: mock_mlx_vlm_speculative_drafters.load_drafter.reset_mock()")

    with open(file_path, "w") as f:
        f.write(content)

    print("File patched successfully.")

patch_file("tests/test_generate_nodes.py")
