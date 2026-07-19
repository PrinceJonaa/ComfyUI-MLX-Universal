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

    def test_predict_happy_path(self):
        mock_execute = MagicMock()
        mock_output = ("out_img", "comb_mask", "ind_masks", '{"data":[]}')
        mock_execute.return_value = mock_output
        self.sam_nodes.execute_sam3_prediction = mock_execute

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

        mock_image = "fake_image_tensor"

        result = node.predict(mock_model, mock_image, "a cat", 0.4)

        self.sam_nodes.execute_sam3_prediction.assert_called_once_with(
            mlx_model=mock_model,
            image=mock_image,
            text_prompt="a cat",
            score_threshold=0.4,
        )
        self.assertEqual(result, ("out_img", "comb_mask", "ind_masks", '{"data":[]}'))


if __name__ == "__main__":
    unittest.main()
