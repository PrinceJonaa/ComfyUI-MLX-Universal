from typing import Any

import torch

from ..runtime.data_types import LoadedMLXModel
from ..runtime.generate_processing import (
    execute_batch_image_description,
    execute_image_description,
    execute_text_generation,
)


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
                "draft_model": ("MLX_DRAFT_MODEL",),
                "enable_thinking": ("BOOLEAN", {"default": False}),
                "thinking_budget": ("INT", {"default": 512, "min": 0, "max": 8192}),
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
        draft_model: Any = None,
        enable_thinking: bool = False,
        thinking_budget: int = 512,
    ) -> tuple:
        if mlx_model.family != "mlx-lm":
            raise ValueError(
                f"Expected model family 'mlx-lm' but found '{mlx_model.family}'. Please ensure you are passing a text model loaded via 'MLX Load Model', not a Vision, Audio, or SAM model."
            )

        response = execute_text_generation(
            mlx_model=mlx_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            draft_model=draft_model,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
        )
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
                "draft_model": ("MLX_DRAFT_MODEL",),
                "draft_kind": (["dflash", "eagle3", "mtp"], {"default": "dflash"}),
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
        image: torch.Tensor | None = None,
        audio_path: str = "",
        draft_model: Any = None,
        draft_kind: str = "dflash",
    ) -> tuple:

        if mlx_model.family != "mlx-vlm":
            raise ValueError(
                f"Expected model family 'mlx-vlm' but found '{mlx_model.family}'. Please ensure you are passing a Vision-Language Model loaded via 'MLX Load Model', not a standard text or SAM model."
            )

        response = execute_image_description(
            mlx_model=mlx_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            image=image,
            audio_path=audio_path,
            draft_model=draft_model,
            draft_kind=draft_kind,
        )
        return (response,)


class MLXBatchVLMDescribeImage:
    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required": {
                "mlx_model": ("MLX_MODEL",),
                "images": ("IMAGE",),
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
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text_list", "concatenated_text")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "run_batch"
    CATEGORY = "MLX Universal/VLM"

    def run_batch(
        self,
        mlx_model: LoadedMLXModel,
        images: torch.Tensor,
        prompt: str,
        max_tokens: int,
        temperature: float,
        seed: int,
        enable_thinking: bool,
        thinking_budget: int,
    ) -> tuple:
        if mlx_model.family != "mlx-vlm":
            raise ValueError(
                f"Expected model family 'mlx-vlm' but found '{mlx_model.family}'. Please ensure you are passing a Vision-Language Model loaded via 'MLX Load Model', not a standard text or SAM model."
            )

        text_list, concatenated = execute_batch_image_description(
            mlx_model=mlx_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            images=images,
        )
        return (text_list, concatenated)


NODE_CLASS_MAPPINGS = {
    "MLXLMGenerateText": MLXLMGenerateText,
    "MLXVLMDescribeImage": MLXVLMDescribeImage,
    "MLXBatchVLMDescribeImage": MLXBatchVLMDescribeImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXLMGenerateText": "MLX Generate Text",
    "MLXVLMDescribeImage": "MLX Understand Image",
    "MLXBatchVLMDescribeImage": "MLX Batch Understand Image",
}
