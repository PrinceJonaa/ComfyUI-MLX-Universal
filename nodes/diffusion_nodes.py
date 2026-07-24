from typing import Any

# DiffusionKit specific imports


class MLXDecoder:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {"required": {"latent_image": ("LATENT",), "mlx_vae": ("mlx_vae",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "MLX Universal/Diffusion"

    def decode(self, latent_image: dict, mlx_vae: Any) -> tuple:
        from ..runtime.diffusion_processing import decode_latents

        return decode_latents(latent_image, mlx_vae)


class MLXSampler:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
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
                        "tooltip": "Classifier-Free Guidance. Base Flux models often use 0, while Schnell/Dev vary.",
                    },
                ),
                "mlx_positive_conditioning": ("mlx_conditioning",),
                "latent_image": ("LATENT",),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Amount of noise to add. 1.0 is full generation from scratch, lower values are for img2img.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate_image"
    CATEGORY = "MLX Universal/Diffusion"

    def generate_image(
        self,
        mlx_model: Any,
        seed: int,
        steps: int,
        cfg: float,
        mlx_positive_conditioning: dict,
        latent_image: dict,
        denoise: float,
    ) -> tuple:
        from ..runtime.diffusion_processing import generate_image

        return generate_image(
            mlx_model=mlx_model,
            seed=seed,
            steps=steps,
            cfg=cfg,
            mlx_positive_conditioning=mlx_positive_conditioning,
            latent_image=latent_image,
            denoise=denoise,
        )


class MLXLoadFlux:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
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

    def load_flux_model(self, model_version: str) -> tuple:
        from ..runtime.model_loader import load_flux_pipeline

        model = load_flux_pipeline(model_version)

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
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "mlx_conditioning": ("mlx_conditioning", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("mlx_conditioning",)
    FUNCTION = "encode"
    CATEGORY = "MLX Universal/Diffusion"

    def encode(self, mlx_conditioning: dict, text: str) -> tuple:
        from ..runtime.diffusion_processing import encode_clip_text

        return encode_clip_text(mlx_conditioning, text)


class MLXEncoder:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {"required": {"image": ("IMAGE",), "mlx_model": ("mlx_model",)}}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_image",)
    FUNCTION = "encode"
    CATEGORY = "MLX Universal/Diffusion"

    def encode(self, image, mlx_model) -> tuple:
        from ..runtime.diffusion_processing import encode_image

        return encode_image(image, mlx_model)


NODE_CLASS_MAPPINGS = {
    "MLXEncoder": MLXEncoder,
    "MLXClipTextEncoder": MLXClipTextEncoder,
    "MLXLoadFlux": MLXLoadFlux,
    "MLXSampler": MLXSampler,
    "MLXDecoder": MLXDecoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXEncoder": "MLX VAE Encode (Flux)",
    "MLXClipTextEncoder": "MLX CLIP Text Encoder",
    "MLXLoadFlux": "MLX Load Flux Model from HF",
    "MLXSampler": "MLX Generate Image (Flux)",
    "MLXDecoder": "MLX VAE Decode (Flux)",
}
