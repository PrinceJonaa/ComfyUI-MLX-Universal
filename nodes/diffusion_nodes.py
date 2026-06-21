import numpy as np
import mlx.core as mx
import torch
import os
from typing import Optional

# DiffusionKit specific imports
from ..diffusionkit.mlx.tokenizer import Tokenizer, T5Tokenizer
from ..diffusionkit.mlx.t5 import SD3T5Encoder
from ..diffusionkit.mlx.clip import CLIPTextModel
from ..diffusionkit.mlx import FluxPipeline
from ..diffusionkit.mlx.constants import T5_MAX_LENGTH


class MLXDecoder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"latent_image": ("LATENT",), "mlx_vae": ("mlx_vae",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "MLX Universal/Diffusion"

    def decode(self, latent_image, mlx_vae):
        from ..runtime.bridge import latent_to_mlx, mlx_to_torch

        mlx_latent = latent_to_mlx(latent_image)
        decoded = mlx_vae(mlx_latent)
        decoded = mx.clip(decoded / 2 + 0.5, 0, 1)
        # Force evaluation here to prevent passing uncomputed graphs to the bridging layer, avoiding deadlocks
        mx.eval(decoded)

        decoded_torch = mlx_to_torch(decoded.astype(mx.float32))
        return (decoded_torch,)


class MLXSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mlx_model": ("mlx_model",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "mlx_positive_conditioning": ("mlx_conditioning",),
                "latent_image": ("LATENT",),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate_image"
    CATEGORY = "MLX Universal/Diffusion"

    def generate_image(
        self,
        mlx_model,
        seed,
        steps,
        cfg,
        mlx_positive_conditioning,
        latent_image,
        denoise,
    ):
        from ..runtime.bridge import mlx_to_latent
        conditioning = mlx_positive_conditioning["conditioning"]
        pooled_conditioning = mlx_positive_conditioning["pooled_conditioning"]

        batch, channels, height, width = latent_image["samples"].shape
        latent_size = (height, width)

        latents, iter_time = mlx_model.denoise_latents(
            conditioning,
            pooled_conditioning,
            num_steps=steps,
            cfg_weight=cfg,
            latent_size=latent_size,
            seed=seed,
            image_path=None,
            denoise=denoise,
        )

        mx.eval(latents)
        latents = latents.astype(mlx_model.activation_dtype)
        return (mlx_to_latent(latents),)


class MLXLoadFlux:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_version": (
                    [
                        "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized",
                        "argmaxinc/mlx-FLUX.1-schnell",
                        "argmaxinc/mlx-FLUX.1-dev",
                    ],
                )
            }
        }

    RETURN_TYPES = ("mlx_model", "mlx_vae", "mlx_conditioning")
    FUNCTION = "load_flux_model"
    CATEGORY = "MLX Universal/Loaders"

    def check_model_folder(self, filename):
        home_dir = os.path.expanduser("~")
        formatted_filename = filename.replace("/", "--")
        folder_path = os.path.join(
            home_dir, ".cache/huggingface/hub/models--" + formatted_filename
        )

        if os.path.exists(folder_path):
            print("Found existing model folder, verifying download...")
        else:
            print("Model folder not found, downloading from HuggingFace... 🤗")

    def load_flux_model(self, model_version):
        self.check_model_folder(model_version)
        from ..runtime.registry import get_or_load_model
        
        def _loader():
            return FluxPipeline(model_version=model_version, low_memory_mode=False, w16=True, a16=True)
            
        model = get_or_load_model(f"flux_{model_version}", _loader)

        clip = {
            "model_name": model_version,
            "clip_l_model": model.clip_l,
            "clip_l_tokenizer": model.tokenizer_l,
            "t5_model": model.t5_encoder,
            "t5_tokenizer": model.t5_tokenizer,
        }

        print("Model successfully loaded.")
        return (model, model.decoder, clip)


class MLXClipTextEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "mlx_conditioning": ("mlx_conditioning", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("mlx_conditioning",)
    FUNCTION = "encode"
    CATEGORY = "MLX Universal/Diffusion"

    def _tokenize(self, tokenizer, text: str, negative_text: Optional[str] = None):
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

    def encode(self, mlx_conditioning, text):
        model_name = mlx_conditioning["model_name"]
        clip_l_encoder: CLIPTextModel = mlx_conditioning["clip_l_model"]
        clip_l_tokenizer: Tokenizer = mlx_conditioning["clip_l_tokenizer"]
        t5_encoder: SD3T5Encoder = mlx_conditioning["t5_model"]
        t5_tokenizer: T5Tokenizer = mlx_conditioning["t5_tokenizer"]

        clip_tokens = self._tokenize(tokenizer=clip_l_tokenizer, text=text)
        clip_l_embeddings = clip_l_encoder(clip_tokens[[0], :])
        clip_pooled_output = clip_l_embeddings.pooled_output

        t5_tokens = self._tokenize(tokenizer=t5_tokenizer, text=text)
        padded_tokens_t5 = mx.zeros((1, T5_MAX_LENGTH[model_name])).astype(
            t5_tokens.dtype
        )
        padded_tokens_t5[:, : t5_tokens.shape[1]] = t5_tokens[[0], :]

        t5_embeddings = t5_encoder(padded_tokens_t5)

        output = {
            "conditioning": t5_embeddings,
            "pooled_conditioning": clip_pooled_output,
        }
        return (output,)


NODE_CLASS_MAPPINGS = {
    "MLXClipTextEncoder": MLXClipTextEncoder,
    "MLXLoadFlux": MLXLoadFlux,
    "MLXSampler": MLXSampler,
    "MLXDecoder": MLXDecoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXClipTextEncoder": "MLX CLIP Text Encoder",
    "MLXLoadFlux": "MLX Load Flux Model from HF",
    "MLXSampler": "MLX Generate Image (Flux)",
    "MLXDecoder": "MLX VAE Decode (Flux)",
}
