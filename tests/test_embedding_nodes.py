import unittest
from unittest.mock import patch

# Ensure mock dependencies
from tests.test_helper import import_node_module


class TestEmbeddingNodes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedding_nodes = import_node_module("embedding_nodes")
        cls.MLXTextEmbedding = cls.embedding_nodes.MLXTextEmbedding

    def test_embedding_node_interface(self):
        node = self.MLXTextEmbedding()
        inputs = node.INPUT_TYPES()
        self.assertIn("text", inputs["required"])
        self.assertIn("model_path", inputs["required"])

    @patch("comfyui_mlx_universal.runtime.embedding_processing.generate_text_embedding")
    def test_embedding_node_execute(self, mock_generate):
        mock_generate.return_value = ("mocked_tensor",)
        node = self.MLXTextEmbedding()
        result = node.generate_embedding("hello world", "model_id")
        self.assertEqual(result, ("mocked_tensor",))
        mock_generate.assert_called_once_with("hello world", "model_id")


if __name__ == "__main__":
    unittest.main()
