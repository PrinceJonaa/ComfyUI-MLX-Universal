from ..runtime.data_types import LoadedMLXModel
from ..runtime.model_loader import (
    apply_dynamic_lora,
    load_draft_model,
    load_unified_mlx_model,
)


class MLXModelLoaderUnified:
    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required": {
                "model_path": (
                    "STRING",
                    {"default": "mlx-community/Qwen3.5-4B-OptiQ-4bit"},
                ),
                "model_type": (
                    ["auto", "mlx-lm", "mlx-vlm", "sam3"],
                    {"default": "auto"},
                ),
                "trust_remote_code": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Required for certain models with custom Python code in their Hugging Face repo.",
                    },
                ),
                "quantize_activations": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Compresses memory during generation. May slightly reduce quality.",
                    },
                ),
            },
            "optional": {
                "adapter_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Optional path to a LoRA adapter directory or Hugging Face repo ID.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MLX_MODEL",)
    RETURN_NAMES = ("mlx_model",)
    FUNCTION = "load_model"
    CATEGORY = "MLX Universal/Loaders"

    def load_model(
        self,
        model_path: str,
        model_type: str,
        trust_remote_code: bool,
        quantize_activations: bool,
        adapter_path: str = "",
    ) -> tuple:
        loaded = load_unified_mlx_model(
            model_path=model_path,
            model_type=model_type,
            trust_remote_code=trust_remote_code,
            quantize_activations=quantize_activations,
            adapter_path=adapter_path,
        )
        return (loaded,)


class MLXApplyLoRA:
    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required": {
                "mlx_model": ("MLX_MODEL",),
                "adapter_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Path to a LoRA adapter directory or Hugging Face repo ID to fuse into the model.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MLX_MODEL",)
    RETURN_NAMES = ("mlx_model",)
    FUNCTION = "apply_lora"
    CATEGORY = "MLX Universal/Loaders"

    def apply_lora(self, mlx_model: LoadedMLXModel, adapter_path: str):
        if not adapter_path:
            return (mlx_model,)

        loaded = apply_dynamic_lora(mlx_model, adapter_path)
        return (loaded,)


class MLXDraftModelLoader:
    @classmethod
    def INPUT_TYPES(s) -> dict:
        return {
            "required": {
                "model_path": (
                    "STRING",
                    {"default": "mlx-community/Qwen2.5-0.5B-Instruct-4bit"},
                ),
                "model_family": (
                    ["mlx-lm", "mlx-vlm"],
                    {"default": "mlx-lm"},
                ),
            }
        }

    RETURN_TYPES = ("MLX_DRAFT_MODEL",)
    RETURN_NAMES = ("draft_model",)
    FUNCTION = "load_model"
    CATEGORY = "MLX Universal/Loaders"

    def load_model(self, model_path: str, model_family: str) -> tuple:
        if not model_path:
            raise ValueError("Draft model path cannot be empty.")

        loaded = load_draft_model(model_path, model_family)
        return (loaded,)


NODE_CLASS_MAPPINGS = {
    "MLXModelLoaderUnified": MLXModelLoaderUnified,
    "MLXApplyLoRA": MLXApplyLoRA,
    "MLXDraftModelLoader": MLXDraftModelLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXModelLoaderUnified": "MLX Load Model",
    "MLXApplyLoRA": "MLX Apply LoRA",
    "MLXDraftModelLoader": "MLX Load Draft Model",
}
