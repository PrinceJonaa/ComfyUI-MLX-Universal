from ..runtime.registry import clear_all_caches, cache_stats

class MLXClearCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    CATEGORY = "MLX Universal/System"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "clear"

    def clear(self):
        before = cache_stats()
        clear_all_caches()
        after = cache_stats()
        return (f"Cleared caches. Before={before['model_count']} models, {before['draft_count']} drafters | After={after['model_count']} models.",)

class MLXCacheStats:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    CATEGORY = "MLX Universal/System"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("stats",)
    FUNCTION = "stats"

    def stats(self):
        stats = cache_stats()
        return (str(stats),)

NODE_CLASS_MAPPINGS = {
    "MLXClearCache": MLXClearCache,
    "MLXCacheStats": MLXCacheStats,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MLXClearCache": "MLX Clear Cache",
    "MLXCacheStats": "MLX Cache Stats",
}
