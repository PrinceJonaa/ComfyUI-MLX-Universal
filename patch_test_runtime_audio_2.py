with open("tests/test_runtime_audio.py", "r") as f:
    content = f.read()

# Fix the import path for mlx_to_torch mock
content = content.replace(
    '@patch("comfyui_mlx_universal.runtime.audio_processing.mlx_to_torch")',
    '@patch("comfyui_mlx_universal.runtime.bridge.mlx_to_torch")',
)

with open("tests/test_runtime_audio.py", "w") as f:
    f.write(content)

print("Fixed tests/test_runtime_audio.py")
