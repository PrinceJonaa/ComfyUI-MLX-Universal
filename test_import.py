import sys
from unittest.mock import MagicMock
class PackageMock(MagicMock):
    pass
mlx = PackageMock()
mlx.core = MagicMock()
mlx.nn = MagicMock()
mlx.utils = MagicMock()
sys.modules['mlx'] = mlx
sys.modules['mlx.core'] = mlx.core
sys.modules['mlx.nn'] = mlx.nn
sys.modules['mlx.utils'] = mlx.utils
sys.modules['mlx_lm'] = MagicMock()
sys.modules['mlx_vlm'] = MagicMock()

argmaxtools = PackageMock()
argmaxtools.test_utils = MagicMock()
argmaxtools.utils = MagicMock()
sys.modules['argmaxtools'] = argmaxtools
sys.modules['argmaxtools.test_utils'] = argmaxtools.test_utils
sys.modules['argmaxtools.utils'] = argmaxtools.utils
sys.modules['cv2'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['sentencepiece'] = MagicMock()
sys.modules['safetensors'] = MagicMock()

beartype = PackageMock()
beartype.typing = MagicMock()
sys.modules['beartype'] = beartype
sys.modules['beartype.typing'] = beartype.typing

sys.modules['regex'] = MagicMock()

import importlib.util

class ComfyMock(MagicMock):
    pass
comfy = ComfyMock()
comfy.utils = MagicMock()
comfy.model_management = MagicMock()
sys.modules['comfy'] = comfy
sys.modules['comfy.utils'] = comfy.utils
sys.modules['comfy.model_management'] = comfy.model_management
sys.modules['folder_paths'] = MagicMock()

spec = importlib.util.spec_from_file_location("comfyui_mlx", "__init__.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["comfyui_mlx"] = mod
try:
    spec.loader.exec_module(mod)
    print("Init successful!")
except Exception as e:
    import traceback
    traceback.print_exc()
