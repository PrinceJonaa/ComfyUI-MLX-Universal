with open("tests/test_runtime_audio.py", "r") as f:
    content = f.read()

# Fix the test placement
old_str = """
if __name__ == "__main__":
    unittest.main()

    @patch("comfyui_mlx_universal.runtime.audio_processing.load_kokoro_pipeline")
    @patch("mlx.core.array")
    @patch("mlx.core.concatenate")
    @patch("mlx.core.eval")
    @patch("comfyui_mlx_universal.runtime.audio_processing.mlx_to_torch")
    def test_execute_kokoro_tts(
        self, mock_mlx_to_torch, mock_eval, mock_concat, mock_array, mock_load
    ):"""

new_str = """
    @patch("comfyui_mlx_universal.runtime.audio_processing.load_kokoro_pipeline")
    @patch("mlx.core.array")
    @patch("mlx.core.concatenate")
    @patch("mlx.core.eval")
    @patch("comfyui_mlx_universal.runtime.audio_processing.mlx_to_torch")
    def test_execute_kokoro_tts(
        self, mock_mlx_to_torch, mock_eval, mock_concat, mock_array, mock_load
    ):"""

content = content.replace(old_str, new_str)

# Add the main block back at the end
content += """
if __name__ == "__main__":
    unittest.main()
"""

with open("tests/test_runtime_audio.py", "w") as f:
    f.write(content)

print("Fixed tests/test_runtime_audio.py")
