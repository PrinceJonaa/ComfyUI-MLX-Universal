import unittest
from unittest.mock import MagicMock, patch
import torch
import os

@unittest.skipIf(
    os.environ.get("REAL_MLX_TESTS") == "1", "Skipping mock tests with real MLX"
)
class TestRuntimeSAM(unittest.TestCase):
    @patch("comfyui_mlx_universal.runtime.sam_processing.process_sam3_result")
    @patch("comfyui_mlx_universal.runtime.sam_processing.tensor_to_pil")
    def test_execute_sam3_prediction(self, mock_tensor_to_pil, mock_process_result):
        from comfyui_mlx_universal.runtime.sam_processing import execute_sam3_prediction

        # Setup mock model
        mock_model = MagicMock()

        # Setup mock image
        mock_pil_image = MagicMock()
        mock_tensor_to_pil.return_value = [mock_pil_image]

        # Setup mock predictor
        mock_predictor_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.scores = [0.9]
        mock_predictor_instance.predict.return_value = mock_result

        # We need to mock the Sam3Predictor class that is instantiated inside the function
        with patch("mlx_vlm.models.sam3.generate.Sam3Predictor", return_value=mock_predictor_instance) as mock_predictor_cls:

            mock_process_result.return_value = ("out_img", "comb_mask", "ind_masks", '{"data":[]}')

            result = execute_sam3_prediction(
                mlx_model=mock_model,
                image=MagicMock(spec=torch.Tensor),
                text_prompt="test prompt",
                score_threshold=0.5
            )

            # Assertions
            mock_predictor_cls.assert_called_once_with(mock_model.model, mock_model.processor, score_threshold=0.5)
            mock_predictor_instance.predict.assert_called_once_with(mock_pil_image, text_prompt="test prompt")
            mock_process_result.assert_called_once_with(mock_result, mock_pil_image)
            self.assertEqual(result, ("out_img", "comb_mask", "ind_masks", '{"data":[]}'))

    @patch("comfyui_mlx_universal.runtime.sam_processing.tensor_to_pil")
    def test_execute_sam3_prediction_empty_image(self, mock_tensor_to_pil):
        from comfyui_mlx_universal.runtime.sam_processing import execute_sam3_prediction

        mock_tensor_to_pil.return_value = []

        with self.assertRaises(ValueError) as context:
            execute_sam3_prediction(
                mlx_model=MagicMock(),
                image=MagicMock(spec=torch.Tensor),
                text_prompt="test",
                score_threshold=0.5
            )

        self.assertIn("Expected an image batch but found empty input", str(context.exception))

if __name__ == "__main__":
    unittest.main()
