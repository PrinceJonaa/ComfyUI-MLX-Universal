import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Create dummy package in sys.modules so the relative import `..runtime.registry` works
# For `nodes.system_nodes` to import `..runtime.registry`, the package must be `comfyui_mlx_universal.nodes` or something.
# The codebase is designed to be dropped into `custom_nodes/comfyui-mlx-universal`.
# This means the parent package is the directory name in `custom_nodes`.
# Let's fake this package structure in sys.modules.

sys.modules['comfyui_mlx_universal'] = MagicMock()
sys.modules['comfyui_mlx_universal.runtime'] = MagicMock()
sys.modules['comfyui_mlx_universal.runtime.registry'] = MagicMock()

import importlib.util

class TestMLXCacheStats(unittest.TestCase):
    def test_stats(self):
        # We need to dynamically load the module with a specific __package__
        spec = importlib.util.spec_from_file_location(
            "comfyui_mlx_universal.nodes.system_nodes",
            os.path.join(os.path.dirname(__file__), '../system_nodes.py')
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["comfyui_mlx_universal.nodes.system_nodes"] = module
        # Mock mlx before executing
        sys.modules['mlx'] = MagicMock()
        sys.modules['mlx.core'] = MagicMock()

        spec.loader.exec_module(module)

        # Now we have the class
        MLXCacheStats = module.MLXCacheStats

        # Test it
        with patch.object(module, 'cache_stats') as mock_cache_stats:
            mock_stats_data = {"model_count": 2, "draft_count": 1}
            mock_cache_stats.return_value = mock_stats_data

            node = MLXCacheStats()
            result = node.stats()

            mock_cache_stats.assert_called_once()
            self.assertEqual(result, (str(mock_stats_data),))

if __name__ == '__main__':
    unittest.main()
