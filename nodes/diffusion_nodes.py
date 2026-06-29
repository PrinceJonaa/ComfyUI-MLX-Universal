import mlx.core as mx
from typing import Optional, Any

# DiffusionKit specific imports
from ..diffusionkit.mlx.tokenizer import Tokenizer, T5Tokenizer
from ..diffusionkit.mlx.t5 import SD3T5Encoder
from ..diffusionkit.mlx.clip import CLIPTextModel
from ..diffusionkit.mlx.constants import T5_MAX_LENGTH


class MLXDecoder:
    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {"required": {"latent_image": ("LATENT",), "mlx_vae": ("mlx_vae",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "MLX Universal/Diffusion"

    def decode(self, latent_image: dict, mlx_vae: Any) -> tuple:
        from ..runtime.bridge import mlx_to_torch, latent_to_mlx

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


class MLXSampler:
    @classmethod
    def INPUT_TYPES(s) -> dict:
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
        from ..runtime.bridge import mlx_to_latent, latent_to_mlx

        try:
            conditioning = mlx_positive_conditioning["conditioning"]
            pooled_conditioning = mlx_positive_conditioning["pooled_conditioning"]
        except (KeyError, TypeError, AttributeError):
            raise ValueError(
                "Expected a valid MLX conditioning dictionary from an MLX CLIP Text Encoder + Invalid or missing conditioning input + Ensure the positive conditioning node is properly connected"
            )

        try:
            batch, channels, height, width = latent_image["samples"].shape
            latent_size = (height, width)
        except (KeyError, TypeError, AttributeError):
            raise ValueError(
                "Expected a valid ComfyUI latent dictionary with a 'samples' tensor + Invalid or missing latent input + Ensure an Empty Latent Image or VAE Encode node is properly connected"
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


class MLXLoadFlux:
    @classmethod
    def INPUT_TYPES(s) -> dict:
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
    def INPUT_TYPES(s) -> dict:
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "mlx_conditioning": ("mlx_conditioning", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("mlx_conditioning",)
    FUNCTION = "encode"
    CATEGORY = "MLX Universal/Diffusion"

    def _tokenize(
        self, tokenizer, text: str, negative_text: Optional[str] = None
    ) -> mx.array:
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

    def encode(self, mlx_conditioning: dict, text: str) -> tuple:
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

        clip_tokens = self._tokenize(tokenizer=clip_l_tokenizer, text=text)
        clip_l_embeddings = clip_l_encoder(clip_tokens[[0], :])
        clip_pooled_output = clip_l_embeddings.pooled_output

        t5_tokens = self._tokenize(tokenizer=t5_tokenizer, text=text)
        t5_tokens_row = t5_tokens[[0], :]
        pad_amount = max(0, int(T5_MAX_LENGTH[model_name]) - int(t5_tokens_row.shape[1]))
        padded_tokens_t5 = mx.pad(
            t5_tokens_row,
            pad_width=((0, 0), (0, pad_amount)),
            constant_values=0
        ).astype(t5_tokens.dtype)

        t5_embeddings = t5_encoder(padded_tokens_t5)

        # Forces simultaneous evaluation to avoid deferred computation slowing down the diffusion loop
        mx.eval(t5_embeddings, clip_pooled_output)
        output = {
            "conditioning": t5_embeddings,
            "pooled_conditioning": clip_pooled_output,
        }
        return (output,)



class MLXEncoder:
    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {"required": {"image": ("IMAGE",), "mlx_model": ("mlx_model",)}}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_image",)
    FUNCTION = "encode"
    CATEGORY = "MLX Universal/Diffusion"

    def encode(self, image, mlx_model) -> tuple:
        from ..runtime.bridge import torch_to_mlx, mlx_to_latent
        import mlx.core as mx

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
