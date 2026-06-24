import os
import json
from huggingface_hub import hf_hub_download
from typing import Any
from .registry import get_or_load_model, get_or_load_draft_model, make_key
from .data_types import LoadedMLXModel


def get_model_config(model_path_or_id: str) -> dict:
    """
    Fetches the `config.json` for a given MLX model path or remote Hugging Face Hub repository.

    If the file exists locally, it reads from the local path. Otherwise, it attempts to download it
    from the Hugging Face Hub. Returns an empty dictionary if the config cannot be found or parsed.
    """
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


def detect_model_type(model_path_or_id: str) -> str:
    """
    Infers the MLX family type ('mlx-lm', 'mlx-vlm', or 'sam3') from a model path or repository ID.

    Examines the `config.json` for architectural hints. If `config.json` is unavailable,
    falls back to heuristic string matching on the model name.
    """
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


def load_unified_mlx_model(
    model_path: str,
    model_type: str,
    trust_remote_code: bool,
    quantize_activations: bool,
    adapter_path: str = "",
) -> LoadedMLXModel:
    """
    Instantiates an MLX model, dynamically loading via mlx-lm or mlx-vlm based on detected type.

    Handles LoRA fusion, remote/local detection, and automatic routing through the unified memory
    cache registry to prevent redundant loading and OOM crashes.
    """
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
        if adapter_path:
            # LoRA weights are fused at load-time rather than dynamically applied to existing instances to ensure safe tracking within the MLX unified memory cache.
            print(
                f"Loading base model '{model_path}' WITH adapter '{adapter_path}' using type mode '{resolved_type}'..."
            )
        else:
            print(f"Loading model '{model_path}' using type mode '{resolved_type}'...")

        kwargs = {"trust_remote_code": trust_remote_code}
        if adapter_path:
            kwargs["adapter_path"] = adapter_path
            if os.path.exists(adapter_path):
                print(f"Applying LoRA adapter from local path: {adapter_path}")
            else:
                print(
                    f"adapter_path '{adapter_path}' not found locally. Assuming remote Hugging Face Hub repo."
                )

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

    return get_or_load_model(cache_key, _load)


def load_draft_model(draft_model_path: str, is_vlm: bool = False) -> Any:
    """
    Loads a draft model for speculative decoding and routes through the draft cache.
    """
    draft_key = make_key(draft_model_path, "draft")

    def _load_draft():
        print(f"Loading draft model '{draft_model_path}'...")
        if is_vlm:
            from mlx_vlm.speculative.drafters import load_drafter
            return load_drafter(draft_model_path)
        else:
            import mlx_lm
            model, _ = mlx_lm.load(draft_model_path)
            return model

    return get_or_load_draft_model(draft_key, _load_draft)


def load_flux_model(model_version: str) -> Any:
    """
    Loads a Flux pipeline model and routes it through the unified memory cache.
    """
    def _loader():
        from ..diffusionkit.mlx import FluxPipeline
        return FluxPipeline(
            model_version=model_version, low_memory_mode=False, w16=True, a16=True
        )

    return get_or_load_model(f"flux_{model_version}", _loader)


def track_audio_model(model_path: str) -> Any:
    """
    Tracks an audio model in the cache registry to trigger memory eviction if needed.
    """
    cache_key = make_key(model_path, "mlx-audio")

    def _loader():
        # mlx-whisper handles its own caching via `_MODEL_CACHE` internally.
        # But to ensure our registry tracking works (which monitors `len(_MODEL_CACHE)` to evict unified memory),
        # we track a placeholder to trigger evictions if needed.
        return True

    return get_or_load_model(cache_key, _loader)
