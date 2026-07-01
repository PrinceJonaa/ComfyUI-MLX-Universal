import torch
from typing import Any, Optional

import mlx.core as mx

from ..diffusionkit.mlx.clip import CLIPTextModel
from ..diffusionkit.mlx.constants import T5_MAX_LENGTH
from ..diffusionkit.mlx.t5 import SD3T5Encoder
from ..diffusionkit.mlx.tokenizer import T5Tokenizer, Tokenizer
from .bridge import latent_to_mlx, mlx_to_latent, mlx_to_torch, torch_to_mlx


def decode_latents(latent_image: dict, mlx_vae: Any) -> tuple:
    """
    Decodes ComfyUI latents into PyTorch images using an MLX VAE.
    """
    mlx_latent = latent_to_mlx(latent_image)

    print("Decoding latent image with MLX VAE...")
    decoded = mlx_vae(mlx_latent)
    decoded = mx.clip(decoded / 2 + 0.5, 0, 1)
    # Force evaluation here to prevent passing uncomputed graphs to the bridging layer, avoiding deadlocks
    mx.eval(decoded)
    print("VAE decoding complete.")

    # Use bridge to convert to PyTorch efficiently
    decoded_torch = mlx_to_torch(decoded.astype(mx.float32))
    return (decoded_torch,)


def generate_image(
    mlx_model: Any,
    seed: int,
    steps: int,
    cfg: float,
    mlx_positive_conditioning: dict,
    latent_image: dict,
    denoise: float,
) -> tuple:
    """
    Generates image latents using an MLX diffusion model.
    """
    try:
        conditioning = mlx_positive_conditioning["conditioning"]
        pooled_conditioning = mlx_positive_conditioning["pooled_conditioning"]
    except (KeyError, TypeError, AttributeError):
        raise ValueError(
            "Expected a valid MLX conditioning dictionary from an MLX CLIP Text Encoder but found an invalid or missing conditioning input. Ensure the positive conditioning node is properly connected."
        )

    try:
        batch, channels, height, width = latent_image["samples"].shape
        latent_size = (height, width)
    except (KeyError, TypeError, AttributeError):
        raise ValueError(
            "Expected a valid ComfyUI latent dictionary with a 'samples' tensor but found an invalid or missing latent input. Ensure an Empty Latent Image or VAE Encode node is properly connected."
        )

    print(f"Generating image latents ({steps} steps)...")
    input_latents = None
    if denoise < 1.0:
        input_latents = latent_to_mlx(latent_image)

    latents, iter_time = mlx_model.denoise_latents(
        conditioning,
        pooled_conditioning,
        num_steps=steps,
        cfg_weight=cfg,
        latent_size=latent_size,
        seed=seed,
        image_path=None,
        denoise=denoise,
        input_latents=input_latents,
    )

    # Evaluates the latent graph lazily accumulated during sampling loops before returning to ComfyUI to prevent upstream deadlock.
    mx.eval(latents)
    print("Latent generation complete.")
    latents = latents.astype(mlx_model.activation_dtype)
    return (mlx_to_latent(latents),)


def _tokenize(tokenizer, text: str, negative_text: Optional[str] = None) -> mx.array:
    if negative_text is None:
        negative_text = ""
    pad_token = tokenizer.eos_token if tokenizer.pad_with_eos else 0

    text = text.replace("’", "'")
    tokens = [tokenizer.tokenize(text)]
    if tokenizer.pad_to_max_length:
        tokens[0].extend([pad_token] * (tokenizer.max_length - len(tokens[0])))
    if negative_text is not None:
        tokens += [tokenizer.tokenize(negative_text)]
    lengths = [len(t) for t in tokens]
    N = max(lengths)
    tokens = [t + [pad_token] * (N - len(t)) for t in tokens]
    return mx.array(tokens)


def encode_clip_text(mlx_conditioning: dict, text: str) -> tuple:
    """
    Encodes text into MLX conditioning tensors using CLIP and T5 models.
    """
    try:
        model_name = mlx_conditioning["model_name"]
        clip_l_encoder: CLIPTextModel = mlx_conditioning["clip_l_model"]
        clip_l_tokenizer: Tokenizer = mlx_conditioning["clip_l_tokenizer"]
        t5_encoder: SD3T5Encoder = mlx_conditioning["t5_model"]
        t5_tokenizer: T5Tokenizer = mlx_conditioning["t5_tokenizer"]
    except (KeyError, TypeError, AttributeError):
        raise ValueError(
            "Expected a valid MLX conditioning dictionary from an MLX Load Flux Model node + Invalid or missing conditioning input + Ensure the MLX Load Flux Model node is properly connected"
        )

    clip_tokens = _tokenize(tokenizer=clip_l_tokenizer, text=text)
    clip_l_embeddings = clip_l_encoder(clip_tokens[[0], :])
    clip_pooled_output = clip_l_embeddings.pooled_output

    t5_tokens = _tokenize(tokenizer=t5_tokenizer, text=text)
    padded_tokens_t5 = mx.zeros((1, T5_MAX_LENGTH.get(model_name, 256))).astype(
        t5_tokens.dtype
    )
    padded_tokens_t5[:, : t5_tokens.shape[1]] = t5_tokens[[0], :]

    t5_embeddings = t5_encoder(padded_tokens_t5)

    # Forces simultaneous evaluation to avoid deferred computation slowing down the diffusion loop
    mx.eval(t5_embeddings, clip_pooled_output)
    output = {
        "conditioning": t5_embeddings,
        "pooled_conditioning": clip_pooled_output,
    }
    return (output,)


def encode_image(image: torch.Tensor, mlx_model: Any) -> tuple:
    """
    Encodes a PyTorch image tensor into ComfyUI latents using an MLX VAE.
    """
    print("Encoding image to latents with MLX VAE...")

    mlx_image = torch_to_mlx(image)
    # Scale ComfyUI PyTorch image tensors from [0, 1] to [-1, 1]
    mlx_image = (mlx_image * 2) - 1.0

    hidden = mlx_model.encoder(mlx_image)
    mean, _ = hidden.split(2, axis=-1)
    latents = mlx_model.latent_format.process_in(mean)

    mx.eval(latents)
    print("VAE encoding complete.")

    return (mlx_to_latent(latents),)
