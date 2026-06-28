from typing import Any
import os
import mlx.core as mx
from ..runtime.data_types import LoadedMLXModel
from ..runtime.bridge import tensor_to_pil
from ..runtime.registry import get_or_load_draft_model, make_key


class MLXLMGenerateText:
    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required": {
                "mlx_model": ("MLX_MODEL",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Explain quantum computing in simple terms.",
                    },
                ),
                "max_tokens": ("INT", {"default": 256, "min": 1, "max": 16384}),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Controls randomness. Lower values are more focused and deterministic, higher values are more creative.",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Nucleus sampling. Only tokens with a cumulative probability above this threshold are considered.",
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "draft_model_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Path to a smaller draft model for speculative decoding speedups.",
                    },
                ),
                "enable_thinking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable thinking tokens for advanced reasoning models.",
                    },
                ),
                "thinking_budget": (
                    "INT",
                    {
                        "default": 512,
                        "min": 0,
                        "max": 8192,
                        "tooltip": "Maximum number of tokens allocated for the model's internal thinking process.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "MLX Universal/LM"

    def generate(
        self,
        mlx_model: LoadedMLXModel,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        draft_model_path: str = "",
        enable_thinking: bool = False,
        thinking_budget: int = 512,
    ) -> tuple:
        if mlx_model.family != "mlx-lm":
            raise ValueError(
                f"Expected model family 'mlx-lm' + Found '{mlx_model.family}' + Please ensure you are passing a text model loaded via 'MLX Load Model', not a Vision, Audio, or SAM model"
            )

        mx.random.seed(seed)
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        tokenizer = mlx_model.processor
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            # tokenize=False ensures the template returns a formatted string instead of token IDs, which mlx_lm.generate expects.
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        # mlx_lm.generate ignores individual kwargs; advanced parameters must be bundled into a sampler object.
        sampler = make_sampler(temp=temperature, top_p=top_p)

        gen_kwargs: dict[str, Any] = {
            "sampler": sampler,
            "max_tokens": max_tokens,
            "verbose": False,
            "enable_thinking": enable_thinking,
            "thinking_budget": thinking_budget,
        }

        if draft_model_path:
            from ..runtime.model_loader import load_draft_model

            draft_model = load_draft_model(draft_model_path, "mlx-lm")
            gen_kwargs["draft_model"] = draft_model

        print(f"Generating text up to {max_tokens} tokens...")
        response = mlx_lm.generate(
            mlx_model.model,
            tokenizer,
            prompt=formatted_prompt,
            **gen_kwargs,
        )
        print("Text generation complete.")
        return (response,)


class MLXVLMDescribeImage:
    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required": {
                "mlx_model": ("MLX_MODEL",),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "Describe this image in detail."},
                ),
                "max_tokens": ("INT", {"default": 256, "min": 1, "max": 16384}),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Controls randomness. Lower values are more focused and deterministic, higher values are more creative.",
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "enable_thinking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable thinking tokens for advanced visual reasoning models like Qwen-VL or LLaVA.",
                    },
                ),
                "thinking_budget": (
                    "INT",
                    {
                        "default": 512,
                        "min": 0,
                        "max": 8192,
                        "tooltip": "Maximum number of tokens allocated for the model's internal thinking process.",
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE",),
                "audio_path": ("STRING", {"default": ""}),
                "draft_model_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Path to a smaller draft model for speculative decoding speedups.",
                    },
                ),
                "draft_kind": (
                    ["dflash", "eagle3", "mtp"],
                    {
                        "default": "dflash",
                        "tooltip": "The speculative decoding algorithm to use (e.g., dflash, eagle3, mtp).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "MLX Universal/VLM"

    def run(
        self,
        mlx_model: LoadedMLXModel,
        prompt: str,
        max_tokens: int,
        temperature: float,
        seed: int,
        enable_thinking: bool,
        thinking_budget: int,
        image: dict | None = None,
        audio_path: str = "",
        draft_model_path: str = "",
        draft_kind: str = "dflash",
    ) -> tuple:

        if mlx_model.family != "mlx-vlm":
            raise ValueError(
                f"Expected model family 'mlx-vlm' + Found '{mlx_model.family}' + Please ensure you are passing a Vision-Language Model loaded via 'MLX Load Model', not a standard text or SAM model"
            )

        mx.random.seed(seed)
        import mlx_vlm
        from mlx_vlm.prompt_utils import apply_chat_template

        pil_images = tensor_to_pil(image) if image is not None else []
        audios = [audio_path] if audio_path and os.path.exists(audio_path) else []

        formatted_prompt = apply_chat_template(
            mlx_model.processor,
            mlx_model.model.config,
            prompt,
            num_images=len(pil_images),
            num_audios=len(audios),
        )

        gen_kwargs: dict[str, Any] = {
            "temp": temperature,
            "max_tokens": max_tokens,
            "verbose": False,
            "enable_thinking": enable_thinking,
            "thinking_budget": thinking_budget,
        }

        if draft_model_path:
            from ..runtime.model_loader import load_draft_model

            draft_model = load_draft_model(draft_model_path, "mlx-vlm")
            gen_kwargs["draft_model"] = draft_model
            gen_kwargs["draft_kind"] = draft_kind

        print(f"Describing image (max {max_tokens} tokens)...")
        response = mlx_vlm.generate(
            mlx_model.model,
            mlx_model.processor,
            formatted_prompt,
            image=pil_images if pil_images else None,
            audio=audios if audios else None,
            **gen_kwargs,
        )
        print("Image description complete.")
        return (response,)


NODE_CLASS_MAPPINGS = {
    "MLXLMGenerateText": MLXLMGenerateText,
    "MLXVLMDescribeImage": MLXVLMDescribeImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXLMGenerateText": "MLX Generate Text",
    "MLXVLMDescribeImage": "MLX Understand Image",
}
