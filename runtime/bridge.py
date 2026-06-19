import numpy as np
import torch
from PIL import Image

def tensor_to_pil(image_tensor: torch.Tensor) -> list[Image.Image]:
    """Convert ComfyUI IMAGE tensor (B, H, W, C) to a list of PIL Images."""
    pil_images = []
    # If the input is in (H, W, C) or similar without batch dim
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    for img in image_tensor:
        # Scale [0, 1] to [0, 255]
        img_np = (img * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        if img_np.ndim == 3 and img_np.shape[2] == 1:
            img_np = np.squeeze(img_np, axis=2)
        pil_images.append(Image.fromarray(img_np))
    return pil_images

def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a ComfyUI IMAGE tensor (1, H, W, C) scaled to [0, 1]."""
    img_np = np.array(pil_image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_np)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(-1)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor

def mlx_to_torch(mlx_tensor) -> torch.Tensor:
    """Convert an MLX array to a ComfyUI IMAGE tensor (B, H, W, C)."""
    import mlx.core as mx
    # MLX is usually (B, H, W, C) but may have different types
    mx.eval(mlx_tensor)
    if mlx_tensor.dtype == mx.bfloat16:
        np_arr = np.array(mlx_tensor.astype(mx.float32))
    else:
        np_arr = np.array(mlx_tensor)
    
    tensor = torch.from_numpy(np_arr)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    return tensor
