with open("tests/test_runtime_audio.py", "r") as f:
    content = f.read()

# Fix the import path for load_kokoro_pipeline mock
content = content.replace(
    '@patch("comfyui_mlx_universal.runtime.audio_processing.load_kokoro_pipeline")',
    '@patch("comfyui_mlx_universal.runtime.model_loader.load_kokoro_pipeline")',
)

with open("tests/test_runtime_audio.py", "w") as f:
    f.write(content)

print("Fixed tests/test_runtime_audio.py")
