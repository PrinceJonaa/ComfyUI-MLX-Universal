import mlx.core as mx
from typing import Any, Callable

_MODEL_CACHE: dict[str, Any] = {}
_DRAFT_CACHE: dict[str, Any] = {}

def make_key(*parts) -> str:
    """Create a string cache key from parameters."""
    return "::".join(str(p) for p in parts)

def get_or_load_model(key: str, loader: Callable[[], Any]) -> Any:
    """Retrieve from cache or load and cache."""
    if key not in _MODEL_CACHE:
        # Prevent runaway memory if cache gets large
        if len(_MODEL_CACHE) >= 2:
            clear_all_caches()
            print("Cleared MLX Metal cache to free unified memory for new model.")
        _MODEL_CACHE[key] = loader()
    return _MODEL_CACHE[key]

def get_or_load_draft_model(key: str, loader: Callable[[], Any]) -> Any:
    """Cache for speculative decoding drafters."""
    if key not in _DRAFT_CACHE:
        _DRAFT_CACHE[key] = loader()
    return _DRAFT_CACHE[key]

def clear_all_caches():
    """Explicitly clear dictionaries and force Metal to release unified memory."""
    before_models = len(_MODEL_CACHE)
    _MODEL_CACHE.clear()
    _DRAFT_CACHE.clear()
    mx.metal.clear_cache()
    return before_models

def cache_stats():
    """Return current cache statistics."""
    return {
        "models": list(_MODEL_CACHE.keys()),
        "draft_models": list(_DRAFT_CACHE.keys()),
        "model_count": len(_MODEL_CACHE),
        "draft_count": len(_DRAFT_CACHE),
    }
