# 🧪 [Testing] Add test for MLXCacheStats.stats method

## Description

🎯 **What:** The testing gap addressed
This PR addresses the lack of test coverage for the `MLXCacheStats.stats` method inside `nodes/system_nodes.py`. Since the method is a wrapper around `runtime.registry.cache_stats`, testing it ensures it correctly interacts with the registry and stringifies the output correctly.

📊 **Coverage:** What scenarios are now tested
- Validates that `cache_stats` from `runtime.registry` is called exactly once when `MLXCacheStats.stats()` is invoked.
- Verifies that the dictionary returned by `cache_stats()` is correctly stringified and returned as a 1-tuple string, matching ComfyUI's custom node expectations.

✨ **Result:** The improvement in test coverage
Test coverage is increased for system nodes, enhancing reliability and confirming that the caching stats functionality remains correctly integrated with `mlx` via the runtime registry. The new unit tests reside in `nodes/tests/test_system_nodes.py` using a mock registry.
