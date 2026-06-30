import re

# Wait, if REAL_MLX_TESTS=1, tests fail.
# How can I test this? I can't really because Apple Silicon runner installs the real MLX and my linux environment cannot run real MLX without crashing.
# But wait... when I ran with `REAL_MLX_TESTS=1` without my `import torch` changes on master, it ALSO failed on my linux VM with:
# "ImportError: libmlx.so: cannot open shared object file"
# But Apple Silicon runner HAS libmlx.so.
# So Apple Silicon test would run successfully if `REAL_MLX_TESTS=1`?
# Wait! In `tests/test_helper.py`:
# `sys.modules["torch"].Tensor = MockTensor`
# I changed it to `setattr(sys.modules["torch"], "Tensor", MockTensor)`.
# But `tests/test_helper.py` only runs this line:
# `if not USE_REAL_MLX: setattr(sys.modules["torch"], "Tensor", MockTensor)`
# So in Apple Silicon CI (`USE_REAL_MLX=1`), it SKIPS this line.
# Meaning it NEVER uses `MockTensor`.
# So what does my `image: torch.Tensor` do?
# In Apple Silicon CI, it imports `torch` at the top of the node files.
# `import torch` works, because `torch` is installed!
# But then the tests run. `nodes/tests/test_*.py` don't exist, only `tests/test_*.py` do.
# Let's look at `tests/test_runtime_bridge.py`. It uses real `torch` if `USE_REAL_MLX=1`.
# Let's look at `tests/test_generate_nodes.py`.
# Wait! In `tests/test_helper.py`:
# `mock_comfy = sys.modules["comfy"]`
# `setattr(mock_comfy, "utils", sys.modules["comfy.utils"])`
# This happens ALWAYS, even if `USE_REAL_MLX=1`!
# Look at `mock_modules` for `comfy`:
# ```python
# mock_modules.extend(["soundfile", "beartype", "sentry_sdk", "folder_paths", "comfy", "comfy.utils", "comfy.model_management"])
# for mod in mock_modules:
#     if mod not in sys.modules:
#         sys.modules[mod] = MagicMock()
# ```
# Wait! The CI runner does NOT install `comfy`. So `comfy` is ALWAYS mocked.
# If `comfy` is ALWAYS mocked, `sys.modules["comfy"]` is a MagicMock.
# My fix changed:
# `mock_comfy.utils = sys.modules["comfy.utils"]`
# to:
# `setattr(mock_comfy, "utils", sys.modules["comfy.utils"])`
# This is completely fine!

# What else did I change?
# `nodes/diffusion_nodes.py`: added `import torch`, changed `image` to `image: torch.Tensor`
# `nodes/generate_nodes.py`: added `import torch`, changed `image` to `image: torch.Tensor | None = None`
# `nodes/video_nodes.py`: added `import torch`, changed `image` to `image: torch.Tensor | None = None`
# `nodes/sam_nodes.py`: added `import torch`, changed `image` to `image: torch.Tensor`

# Wait! Does `import torch` cause a problem in CI?
# The error was "ModuleNotFoundError: No module named 'torch'" earlier in my tests.
# If `torch` wasn't installed, the Apple Silicon CI WOULD throw "ModuleNotFoundError: No module named 'torch'".
# But the Apple Silicon CI `.github/workflows/mlx_test.yml` clearly has:
# `pip install torch torchvision torchaudio`
# BUT wait! Look at the pip install command:
# `pip install torch torchvision torchaudio`
# On macOS, pip installing torch sometimes installs the CPU version, which is perfectly fine.
# Could the CI have failed because of a typing error?
# My fix: `def encode(self, image: torch.Tensor, mlx_model: Any) -> tuple:`
# Wait! In `tests/test_diffusion_nodes.py` or similar, they might call `encode()`!
