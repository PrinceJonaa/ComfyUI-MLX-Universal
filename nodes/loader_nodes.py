import os
import json
from huggingface_hub import hf_hub_download
from ..runtime.registry import get_or_load_model, make_key
from ..runtime.data_types import LoadedMLXModel


def get_model_config(model_path_or_id):
    if os.path.exists(os.path.join(model_path_or_id, "config.json")):
        config_path = os.path.join(model_path_or_id, "config.json")
    else:
        try:
            config_path = hf_hub_download(
                repo_id=model_path_or_id, filename="config.json"
            )
        except Exception:
            return {}
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def detect_model_type(model_path_or_id):
    config = get_model_config(model_path_or_id)
    if not config:
        name_lower = model_path_or_id.lower()
        if "sam3" in name_lower:
            return "sam3"
        if any(v in name_lower for v in ["vlm", "vl-", "llava", "gemma-3n", "gemma-4"]):
            return "mlx-vlm"
        return "mlx-lm"
    model_type = config.get("model_type", "").lower()
    architectures = [a.lower() for a in config.get("architectures", [])]
    if (
        "sam3" in model_type
        or any("sam3" in arch for arch in architectures)
        or "sam3" in model_path_or_id.lower()
    ):
        return "sam3"
    vlm_indicators = [
        "vlm",
        "vl",
        "llava",
        "idefics",
        "mflux",
        "paligemma",
        "clip",
        "siglip",
        "gemma_3n",
        "gemma_4",
    ]
    is_vlm = (
        any(ind in model_type for ind in vlm_indicators)
        or any(any(ind in arch for ind in vlm_indicators) for arch in architectures)
        or "vision_config" in config
    )
    if is_vlm:
        return "mlx-vlm"
    return "mlx-lm"


class MLXModelLoaderUnified:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": "mlx-community/Qwen3.5-4B-OptiQ-4bit"}),
                "model_type": (["auto", "mlx-lm", "mlx-vlm", "sam3"], {"default": "auto"}),
                "trust_remote_code": ("BOOLEAN", {"default": False}),
                "quantize_activations": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "adapter_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MLX_MODEL",)
    RETURN_NAMES = ("mlx_model",)
    FUNCTION = "load_model"
    CATEGORY = "MLX Universal/Loaders"

    def load_model(
        self,
        model_path,
        model_type,
        trust_remote_code,
        quantize_activations,
        adapter_path="",
    ):
        resolved_type = (
            detect_model_type(model_path) if model_type == "auto" else model_type
        )
        print(
            f"Automatically detected model type for '{model_path}': {resolved_type}"
            if model_type == "auto"
            else f"Using specified model type: {resolved_type}"
        )

        key_suffix = f":adapter={adapter_path}" if adapter_path else ""
        cache_key = (
            make_key(model_path, resolved_type, trust_remote_code, quantize_activations)
            + key_suffix
        )

        def _load():
            print(f"Loading model '{model_path}' using type mode '{resolved_type}'...")
            kwargs = {"trust_remote_code": trust_remote_code}
            if adapter_path:
                kwargs["adapter_path"] = adapter_path
                if os.path.exists(adapter_path):
                    print(f"Applying LoRA adapter from local path: {adapter_path}")
                else:
                    print(f"adapter_path '{adapter_path}' not found locally. Assuming remote Hugging Face Hub repo.")
            if resolved_type == "mlx-lm":
                import mlx_lm

                model, processor = mlx_lm.load(model_path, **kwargs)
                return LoadedMLXModel(
                    family=resolved_type,
                    model_path=model_path,
                    model_type=resolved_type,
                    trust_remote_code=trust_remote_code,
                    quantize_activations=quantize_activations,
                    model=model,
                    processor=processor,
                )
            elif resolved_type in ("mlx-vlm", "sam3"):
                import mlx_vlm

                kwargs["quantize_activations"] = quantize_activations
                model, processor = mlx_vlm.load(model_path, **kwargs)
                return LoadedMLXModel(
                    family=resolved_type,
                    model_path=model_path,
                    model_type=resolved_type,
                    trust_remote_code=trust_remote_code,
                    quantize_activations=quantize_activations,
                    model=model,
                    processor=processor,
                )
            else:
                raise ValueError(f"Unknown resolved model type: {resolved_type}")

        loaded = get_or_load_model(cache_key, _load)
        return (loaded,)


class MLXApplyLoRA:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mlx_model": ("MLX_MODEL",),
                "adapter_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MLX_MODEL",)
    RETURN_NAMES = ("mlx_model",)
    FUNCTION = "apply_lora"
    CATEGORY = "MLX Universal/Loaders"

    def apply_lora(self, mlx_model: LoadedMLXModel, adapter_path: str):
        if not adapter_path:
            return (mlx_model,)

        print(f"Intercepting MLX Model payload to fuse LoRA adapter: {adapter_path}")

        resolved_type = mlx_model.model_type
        model_path = mlx_model.model_path
        trust_remote_code = mlx_model.trust_remote_code
        quantize_activations = mlx_model.quantize_activations

        key_suffix = f":adapter={adapter_path}"
        cache_key = (
            make_key(model_path, resolved_type, trust_remote_code, quantize_activations)
            + key_suffix
        )

        def _load():
            # LoRA weights are fused at load-time rather than dynamically applied to existing instances to ensure safe tracking within the unified memory cache.
            print(f"Loading base model '{model_path}' WITH adapter '{adapter_path}'...")
            kwargs = {"trust_remote_code": trust_remote_code}
            if adapter_path:
                kwargs["adapter_path"] = adapter_path
                if not os.path.exists(adapter_path):
                    print(f"adapter_path '{adapter_path}' not found locally. Assuming remote Hugging Face Hub repo.")
            if resolved_type == "mlx-lm":
                import mlx_lm

                model, processor = mlx_lm.load(model_path, **kwargs)
                return LoadedMLXModel(
                    family=resolved_type,
                    model_path=model_path,
                    model_type=resolved_type,
                    trust_remote_code=trust_remote_code,
                    quantize_activations=quantize_activations,
                    model=model,
                    processor=processor,
                )
            elif resolved_type in ("mlx-vlm", "sam3"):
                import mlx_vlm

                kwargs["quantize_activations"] = quantize_activations
                model, processor = mlx_vlm.load(model_path, **kwargs)
                return LoadedMLXModel(
                    family=resolved_type,
                    model_path=model_path,
                    model_type=resolved_type,
                    trust_remote_code=trust_remote_code,
                    quantize_activations=quantize_activations,
                    model=model,
                    processor=processor,
                )
            else:
                raise ValueError(f"Unknown resolved model type: {resolved_type}")

        loaded = get_or_load_model(cache_key, _load)
        return (loaded,)


NODE_CLASS_MAPPINGS = {
    "MLXModelLoaderUnified": MLXModelLoaderUnified,
    "MLXApplyLoRA": MLXApplyLoRA,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXModelLoaderUnified": "MLX Load Model",
    "MLXApplyLoRA": "MLX Apply LoRA",
}
