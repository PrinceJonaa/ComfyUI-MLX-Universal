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
    # Prevents silent data corruption or performance bottlenecks when PyTorch accesses uncomputed lazy arrays
    mx.eval(mlx_tensor)
    if mlx_tensor.dtype == mx.bfloat16:
        mlx_tensor = mlx_tensor.astype(mx.float32)
        # Explicit evaluation required because type casting creates a new uncomputed lazy array
        mx.eval(mlx_tensor)
        np_arr = np.array(mlx_tensor)
    else:
        np_arr = np.array(mlx_tensor)

    tensor = torch.from_numpy(np_arr)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    return tensor


def torch_to_mlx(torch_tensor: torch.Tensor):
    """Convert a PyTorch tensor to an MLX array."""
    import mlx.core as mx

    # Move to CPU and numpy before creating MLX array
    # Avoids conversion errors with tensors requiring gradients or residing on a GPU
    np_arr = torch_tensor.cpu().detach().numpy()
    mlx_tensor = mx.array(np_arr)
    return mlx_tensor


def mlx_to_latent(mlx_tensor) -> dict:
    """Convert an MLX latent array into a ComfyUI-compliant latent dictionary."""
    import mlx.core as mx

    # Explicit evaluation prevents latent structures from retaining uncomputed graphs that deadlock the PyTorch backend.
    mx.eval(mlx_tensor)
    if mlx_tensor.dtype == mx.bfloat16:
        mlx_tensor = mlx_tensor.astype(mx.float32)
        # Explicit evaluation required because type casting creates a new uncomputed lazy array
        mx.eval(mlx_tensor)
        np_arr = np.array(mlx_tensor)
    else:
        np_arr = np.array(mlx_tensor)
    torch_tensor = torch.from_numpy(np_arr)
    # ComfyUI natively expects latents to be packaged as a dictionary with a 'samples' key rather than a raw tensor.
    return {"samples": torch_tensor}


def latent_to_mlx(latent_dict: dict):
    """Safely unpack a ComfyUI latent dictionary and convert its PyTorch tensor back to an MLX array."""
    if not isinstance(latent_dict, dict) or "samples" not in latent_dict:
        raise ValueError(
            "Expected a ComfyUI latent dictionary with a 'samples' key but found an invalid or missing latent format. Ensure a properly formatted latent image is provided."
        )
    return torch_to_mlx(latent_dict["samples"])
