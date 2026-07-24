# Contributing to ComfyUI MLX Universal Node Pack

First off, thank you for considering contributing! This project is designed as a foundational **base layer** for the ComfyUI and Apple MLX communities. 

To keep this repository scalable and prevent Out-Of-Memory (OOM) spaghetti code, we enforce a strict architectural separation known as the **Runtime Substrate**. 

## The Golden Rules

Before you open a Pull Request to add a new model family (e.g., Audio, VAEs, new Diffusion samplers), you **MUST** adhere to the following structure:

### 1. Do Not Put Logic in the UI Nodes
The `nodes/` folder is strictly for ComfyUI's frontend wrappers (e.g., `INPUT_TYPES`, `RETURN_TYPES`). Do not put heavy generation loops, file downloading logic, or massive tensor slicing directly in these classes. 

### 2. Always Use the Cache Registry
Apple Silicon uses Unified Memory. If you load a 14B VLM without tracking it, you will crash the user's computer. 
**DO NOT** do this:
```python
# BAD
import mlx_vlm

model, processor = mlx_vlm.load("mlx-community/model")
```
**DO** this:
```python
# GOOD
from ..runtime.registry import get_or_load_model


def _loader():
    return mlx_vlm.load("model_path")


loaded = get_or_load_model("my_unique_key", _loader)
```

### 3. Always Use the Tensor Bridge
ComfyUI speaks PyTorch (`[B, H, W, C]`). MLX speaks `mlx.core.array`. To prevent shape mismatches and lazy evaluation traps, all outputs must be routed through `runtime/bridge.py`.
**DO NOT** return `mlx.core.array` directly from a node! ComfyUI will not know how to handle it.

### 4. Explicit Evaluation
MLX is lazy. If your node computes a mask or an image, explicitly call `mx.eval(your_array)` before converting it to PyTorch/Numpy, otherwise you will cause severe lag or crashes in downstream ComfyUI nodes.

## How to Add a New Modality
1. Create a new file in the `nodes/` directory (e.g., `nodes/audio_nodes.py`).
2. Write your thin wrapper node.
3. Import your new mappings into the root `__init__.py` file dynamically.
4. If your modality introduces new data payloads, add a `@dataclass` to `runtime/data_types.py`.
