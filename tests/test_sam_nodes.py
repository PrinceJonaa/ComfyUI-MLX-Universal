import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

from tests.test_helper import import_node_module


@unittest.skipIf(
    os.environ.get("REAL_MLX_TESTS") == "1", "Skipping mock tests with real MLX"
)
class TestSAMNodes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sam_nodes = import_node_module("sam_nodes")
        cls.MLXSAM3Predictor = cls.sam_nodes.MLXSAM3Predictor

        runtime_data_types = sys.modules["comfyui_mlx_universal.runtime.data_types"]
        cls.LoadedMLXModel = runtime_data_types.LoadedMLXModel

    def test_predict_wrong_model_family_raises_value_error(self):
        node = self.MLXSAM3Predictor()

        mock_model = self.LoadedMLXModel(
            family="mlx-lm",  # not sam3
            model_path="original/path",
            model_type="mlx-lm",
            trust_remote_code=False,
            quantize_activations=True,
            model=MagicMock(),
            processor=MagicMock(),
        )

        with self.assertRaises(ValueError) as context:
            node.predict(mock_model, {}, "dog", 0.3)
        self.assertIn("Expected model family 'sam3'", str(context.exception))

    @patch("comfyui_mlx_universal.runtime.sam_processing.execute_sam3_prediction")
    def test_predict_happy_path(self, mock_execute):
        mock_execute.return_value = (
            "out_img",
            "comb_mask",
            "ind_masks",
            '{"data":[]}',
        )
        node = self.MLXSAM3Predictor()

        mock_model = self.LoadedMLXModel(
            family="sam3",
            model_path="original/path",
            model_type="sam3",
            trust_remote_code=False,
            quantize_activations=False,
            model="internal_model",
            processor="internal_processor",
        )

        fake_tensor = torch.zeros((1, 3, 3))
        result = node.predict(mock_model, fake_tensor, "a cat", 0.4)

        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], "out_img")
        self.assertEqual(result[3], '{"data":[]}')
        mock_execute.assert_called_once_with(mock_model, fake_tensor, "a cat", 0.4)


if __name__ == "__main__":
    unittest.main()
