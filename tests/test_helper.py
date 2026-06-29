import importlib.util
import os
import sys
from unittest.mock import MagicMock

# Add the project root to sys.path so relative files can be located
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


# Define a MockTensor class so isinstance(x, torch.Tensor) does not crash
class MockTensor(MagicMock):
    pass


# 1. Pre-mock all external dependencies that might not exist in the test environment
# We also include intermediate parent packages so Python's import system resolves them correctly
mock_modules = [
    "mlx",
    "mlx.core",
    "mlx.nn",
    "mlx.optimizers",
    "mlx_lm",
    "mlx_lm.sample_utils",
    "mlx_vlm",
    "mlx_vlm.prompt_utils",
    "mlx_vlm.speculative",
    "mlx_vlm.speculative.drafters",
    "mlx_vlm.models",
    "mlx_vlm.models.sam3",
    "mlx_vlm.models.sam3.generate",
    "mlx_video",
    "mlx_whisper",
    "soundfile",
    "beartype",
    "sentry_sdk",
    "folder_paths",
    "comfy",
    "comfy.utils",
    "comfy.model_management",
    "numpy",
    "huggingface_hub",
    "torch",
    "torch.nn",
    "torchvision",
    "torchvision.transforms",
    "torchaudio",
    "safetensors",
    "safetensors.torch",
    "PIL",
    "PIL.Image",
    "mlx_audio",
    "mlx_audio.tts",
    "mlx_audio.tts.models",
    "mlx_audio.tts.models.kokoro",
]

for mod in mock_modules:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Inject MockTensor into mocked torch
sys.modules["torch"].Tensor = MockTensor

# Setup default mocks for comfyui components to avoid crashes
mock_comfy = sys.modules["comfy"]
mock_comfy.utils = sys.modules["comfy.utils"]
mock_comfy.model_management = sys.modules["comfy.model_management"]

mock_folder_paths = sys.modules["folder_paths"]
mock_folder_paths.get_temp_directory.return_value = "/tmp"

# 2. Programmatically register comfyui_mlx_universal as a package in sys.modules
# This allows relative imports (e.g. from ..runtime.registry import ...) to resolve correctly
if "comfyui_mlx_universal" not in sys.modules:
    sys.modules["comfyui_mlx_universal"] = MagicMock()
if "comfyui_mlx_universal.runtime" not in sys.modules:
    sys.modules["comfyui_mlx_universal.runtime"] = MagicMock()
if "comfyui_mlx_universal.diffusionkit" not in sys.modules:
    sys.modules["comfyui_mlx_universal.diffusionkit"] = MagicMock()
if "comfyui_mlx_universal.diffusionkit.mlx" not in sys.modules:
    sys.modules["comfyui_mlx_universal.diffusionkit.mlx"] = MagicMock()

# Mock submodules of diffusionkit.mlx so they can be imported directly
diffusionkit_submodules = ["tokenizer", "t5", "clip", "constants"]
for sub in diffusionkit_submodules:
    fullname = f"comfyui_mlx_universal.diffusionkit.mlx.{sub}"
    if fullname not in sys.modules:
        sys.modules[fullname] = MagicMock()


def import_submodule(subfolder, name):
    """Dynamically registers a module from the subfolder inside the package namespace."""
    full_name = f"comfyui_mlx_universal.{subfolder}.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    file_path = os.path.join(root_dir, subfolder, f"{name}.py")
    if not os.path.exists(file_path):
        # Could be a directory package
        file_path = os.path.join(root_dir, subfolder, name, "__init__.py")

    spec = importlib.util.spec_from_file_location(full_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        # If compilation fails due to mock objects, we print a warning but keep the module object
        print(f"Warning: could not execute {full_name}: {e}")
    return module


# Pre-import key runtime files so nodes can resolve them relative to their package parent
try:
    import_submodule("runtime", "data_types")
    import_submodule("runtime", "registry")
    import_submodule("runtime", "bridge")
    import_submodule("runtime", "model_loader")
    import_submodule("runtime", "video_processing")
    import_submodule("runtime", "sam_processing")
    import_submodule("runtime", "audio_processing")
except Exception as e:
    print(f"Warning during pre-import: {e}")


def import_node_module(node_file_basename):
    """
    Loads a node module under the comfyui_mlx_universal.nodes package namespace
    so that relative imports resolve correctly.
    """
    module_name = f"comfyui_mlx_universal.nodes.{node_file_basename}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    file_path = os.path.join(root_dir, "nodes", f"{node_file_basename}.py")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
