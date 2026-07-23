import os
import sys
import unittest
from unittest.mock import patch

# Add root directory to sys.path to find tests package
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from tests.test_helper import import_node_module  # noqa: E402


class TestMLXCacheStats(unittest.TestCase):
    def test_stats(self):
        # Dynamically import system_nodes using our test helper
        system_nodes = import_node_module("system_nodes")
        mlx_cache_stats = system_nodes.MLXCacheStats

        # Patch cache_stats in the imported module namespace
        with patch.object(system_nodes, "cache_stats") as mock_cache_stats:
            mock_stats_data = {"model_count": 2, "draft_count": 1}
            mock_cache_stats.return_value = mock_stats_data

            node = mlx_cache_stats()
            result = node.stats()

            mock_cache_stats.assert_called_once()
            self.assertEqual(result, (str(mock_stats_data),))


if __name__ == "__main__":
    unittest.main()
