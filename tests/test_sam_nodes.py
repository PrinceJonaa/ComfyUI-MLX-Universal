import sys
import unittest
from unittest.mock import MagicMock

from tests.test_helper import import_node_module


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
        # Configure the Sam3Predictor mock registered in sys.modules
        mock_sam_predictor_cls = sys.modules["mlx_vlm.models.sam3.generate"].Sam3Predictor
        mock_sam_predictor_cls.reset_mock()

        mock_predictor = MagicMock()
        mock_sam_predictor_cls.return_value = mock_predictor

        mock_detection_result = MagicMock()
        mock_detection_result.scores = [0.9, 0.8]
        mock_predictor.predict.return_value = mock_detection_result

        # Configure clean manual mock for process_sam3_result in the sam_nodes module
        mock_output = ("out_img", "comb_mask", "ind_masks", '{"data":[]}')
        self.sam_nodes.process_sam3_result = MagicMock(return_value=mock_output)

        # Setup mock node
        node = self.MLXSAM3Predictor()

        # Setup clean mock for tensor_to_pil in the sam_nodes module
        mock_pil_img = MagicMock()
        mock_pil_img.size = (512, 512)
        self.sam_nodes.tensor_to_pil = MagicMock(return_value=[mock_pil_img])

        mock_model = self.LoadedMLXModel(
            family="sam3",
            model_path="original/path",
            model_type="sam3",
            trust_remote_code=False,
            quantize_activations=False,
            model="internal_model",
            processor="internal_processor",
        )

        result = node.predict(mock_model, "fake_image_tensor", "a cat", 0.4)

        # Asserts
        self.sam_nodes.tensor_to_pil.assert_called_once_with("fake_image_tensor")
        mock_sam_predictor_cls.assert_called_once_with(
            "internal_model", "internal_processor", score_threshold=0.4
        )
        mock_predictor.predict.assert_called_once_with(
            mock_pil_img, text_prompt="a cat"
        )
        self.sam_nodes.process_sam3_result.assert_called_once_with(
            mock_detection_result, mock_pil_img
        )
        self.assertEqual(result, ("out_img", "comb_mask", "ind_masks", '{"data":[]}'))


if __name__ == "__main__":
    unittest.main()
