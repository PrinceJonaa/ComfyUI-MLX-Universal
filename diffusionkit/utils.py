import numpy as np
from PIL import Image


def bytes2gigabytes(n: int) -> int:
    """Convert bytes to gigabytes"""
    return int(n / 1024**3)


def image_psnr(reference: Image, proxy: Image) -> float:
    """Peak-Signal-to-Noise-Ratio in dB between a reference
    and a proxy PIL.Image
    """
    reference = np.asarray(reference)
    proxy = np.asarray(proxy)

    assert reference.squeeze().shape == proxy.squeeze().shape, (
        f"{reference.shape} is incompatible with {proxy.shape}!"
    )
    reference = reference.flatten()
    proxy = proxy.flatten()

    peak_signal = np.abs(reference).max()
    mse = np.sqrt(np.mean((reference - proxy) ** 2))
    return 20 * np.log10((peak_signal + 1e-5) / (mse + 1e-10))


def compute_psnr(reference: np.array, proxy: np.array) -> float:
    """Peak-Signal-to-Noise-Ratio in dB between a reference
    and a proxy np.array
    """
    assert reference.squeeze().shape == proxy.squeeze().shape, (
        f"{reference.shape} is incompatible with {proxy.shape}!"
    )
    reference = reference.flatten()
    proxy = proxy.flatten()

    peak_signal = np.abs(reference).max()
    mse = np.sqrt(np.mean((reference - proxy) ** 2))
    return 20 * np.log10((peak_signal + 1e-5) / (mse + 1e-10))


def _load_weights(module, pt_path: str):
    import torch

    state_dict = torch.load(pt_path)
    total_params = sum(p.numel() for p in module.parameters())
    state_dict_params = sum(p.numel() for p in state_dict.values())

    if total_params != state_dict_params:
        raise ValueError(
            f"Total number of parameters in state_dict ({state_dict_params}) does not match the number of parameters in the module ({total_params})"
        )
    module.load_state_dict(state_dict)
