import copy
import json
import os

from huggingface_hub import hf_hub_download

from .data_types import LoadedMLXModel
from .registry import get_or_load_model, make_key


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
            raise ValueError(
                f"Expected a known resolved model type (e.g. 'mlx-lm', 'mlx-vlm', 'sam3') + Found '{resolved_type}' + Check the model path or force the model type manually"
            )

    return get_or_load_model(cache_key, _load)


def apply_dynamic_lora(base_model: LoadedMLXModel, adapter_path: str) -> LoadedMLXModel:
    """
    Dynamically fuses a LoRA adapter into an existing loaded model using a deepcopy of the model graph,
    avoiding redundant disk I/O and base model instantiation. Tracks the result via the registry.
    """
    if not adapter_path:
        return base_model

    print(f"Intercepting MLX Model payload to dynamically fuse LoRA adapter: {adapter_path}")

    # Use the same cache key construction but inject the adapter suffix to properly track the new state
    key_suffix = f":adapter={adapter_path}"
    cache_key = (
        make_key(
            base_model.model_path,
            base_model.model_type,
            base_model.trust_remote_code,
            base_model.quantize_activations,
        )
        + key_suffix
    )

    def _fuse_lora():
        print(f"Dynamically applying LoRA weights from '{adapter_path}' to model '{base_model.model_path}'...")
        # Deepcopy the parameter graph so we don't permanently alter the cached base model for other nodes
        model_copy = copy.deepcopy(base_model.model)

        if base_model.family == "mlx-lm":
            from mlx_lm.utils import load_adapters
            model_copy = load_adapters(model_copy, adapter_path)
            model_copy.eval()
        elif base_model.family in ("mlx-vlm", "sam3"):
            from mlx_vlm.trainer.utils import apply_lora_layers
            model_copy = apply_lora_layers(model_copy, adapter_path)
            model_copy.eval()
        else:
            raise ValueError(f"Dynamic LoRA fusion is not supported for family: {base_model.family}")

        return LoadedMLXModel(
            family=base_model.family,
            model_path=base_model.model_path,
            model_type=base_model.model_type,
            trust_remote_code=base_model.trust_remote_code,
            quantize_activations=base_model.quantize_activations,
            model=model_copy,
            processor=base_model.processor,
        )

    return get_or_load_model(cache_key, _fuse_lora)


def load_draft_model(draft_model_path: str, model_type: str):
    """
    Loads a draft model for speculative decoding and tracks it in the draft cache.
    """
    from .registry import get_or_load_draft_model, make_key

    draft_key = make_key(draft_model_path, "draft")

    def _load_draft():
        print(f"Loading draft model '{draft_model_path}'...")
        if model_type == "mlx-lm":
            import mlx_lm

            model, _ = mlx_lm.load(draft_model_path)
            return model
        elif model_type == "mlx-vlm":
            from mlx_vlm.speculative.drafters import load_drafter

            return load_drafter(draft_model_path)
        else:
            raise ValueError(f"Unsupported draft model type: {model_type}")

    return get_or_load_draft_model(draft_key, _load_draft)


def track_audio_model(model_path: str):
    """
    Dummy load function to ensure MLX Whisper models are tracked by the cache registry
    so that memory eviction is triggered correctly.
    """
    from .registry import get_or_load_model, make_key

    cache_key = make_key(model_path, "mlx-audio")

    def _loader():
        return True

    return get_or_load_model(cache_key, _loader)


def load_flux_pipeline(model_version: str):
    """
    Loads a Flux Pipeline model, downloading it if necessary, and tracks it via the registry.
    """
    import os

    home_dir = os.path.expanduser("~")
    formatted_filename = model_version.replace("/", "--")
    folder_path = os.path.join(
        home_dir, ".cache/huggingface/hub/models--" + formatted_filename
    )

    if os.path.exists(folder_path):
        print("Found existing model folder, verifying download...")
    else:
        print("Model folder not found, downloading from HuggingFace... 🤗")

    # Lazy import to keep MLX separation
    from ..diffusionkit.mlx import FluxPipeline
    from .registry import get_or_load_model

    def _loader():
        return FluxPipeline(
            model_version=model_version, low_memory_mode=False, w16=True, a16=True
        )

    return get_or_load_model(f"flux_{model_version}", _loader)


def load_kokoro_pipeline(model_version: str):
    """
    Loads a Kokoro TTS pipeline, and tracks it via the registry to ensure unified memory is managed correctly.
    """
    # Lazy import to keep MLX separation
    from mlx_audio.tts.models.kokoro import KokoroModel, KokoroPipeline

    from .registry import get_or_load_model

    def _loader():
        print(f"Loading Kokoro TTS model '{model_version}'...")
        # A pipeline uses 'a' (American) or 'b' (British).
        # We can default to 'a' as it covers most voices and voice styles can be managed by the node input.
        model = KokoroModel(repo_id=model_version)
        pipeline = KokoroPipeline(lang_code="a", model=model)
        return pipeline

    return get_or_load_model(f"kokoro_{model_version}", _loader)


def load_embedding_model(model_path: str):
    """
    Loads a text embedding model via mlx-embeddings, tracking it in the registry to manage memory.
    """
    from .registry import get_or_load_model, make_key

    cache_key = make_key(model_path, "mlx-embeddings")

    def _loader():
        print(f"Loading embedding model '{model_path}'...")
        from mlx_embeddings.utils import load

        model, tokenizer = load(model_path)
        return model, tokenizer

    return get_or_load_model(cache_key, _loader)
