# Removed unused mock assignments for mlx dependencies
# as they are now handled by test_runtime_generate.py
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from tests.test_helper import import_node_module


@unittest.skipIf(
    os.environ.get("REAL_MLX_TESTS") == "1", "Skipping mock tests with real MLX"
)
class TestGenerateNodes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Programmatically import generate_nodes using test_helper
        cls.generate_nodes = import_node_module("generate_nodes")
        cls.MLXLMGenerateText = cls.generate_nodes.MLXLMGenerateText
        cls.MLXVLMDescribeImage = cls.generate_nodes.MLXVLMDescribeImage

        # We need to mock runtime classes used in tests
        runtime_data_types = sys.modules["comfyui_mlx_universal.runtime.data_types"]
        cls.LoadedMLXModel = runtime_data_types.LoadedMLXModel

    def setUp(self):
        pass

    def get_mocked_model(self):
        model = self.LoadedMLXModel(
            family="mlx-lm",
            model_path="mock_path",
            model_type="mlx-lm",
            trust_remote_code=False,
            quantize_activations=False,
            model=MagicMock(),
            processor=MagicMock(),
        )
        return model

    # --- MLXLMGenerateText Tests ---

    def test_mlx_lm_generate_unknown_model_family_raises_value_error(self):
        node = self.MLXLMGenerateText()
        mocked_model = self.get_mocked_model()
        mocked_model.family = "unknown-family"

        with self.assertRaises(ValueError) as context:
            node.generate(
                mlx_model=mocked_model,
                prompt="test prompt",
                max_tokens=100,
                temperature=0.7,
                top_p=0.9,
                seed=42,
                draft_model=None,
                enable_thinking=False,
                thinking_budget=512,
            )

        self.assertEqual(
            str(context.exception),
            "Expected model family 'mlx-lm' but found 'unknown-family'. Please ensure you are passing a text model loaded via 'MLX Load Model', not a Vision, Audio, or SAM model.",
        )

    @patch.object(import_node_module("generate_nodes"), "execute_text_generation")
    def test_mlx_lm_generate_happy_path_with_chat_template(self, mock_execute):
        mock_execute.return_value = "generated response"
        node = self.MLXLMGenerateText()
        mocked_model = self.get_mocked_model()
        mocked_model.family = "mlx-lm"

        result = node.generate(
            mlx_model=mocked_model,
            prompt="Hello",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            seed=42,
            draft_model=None,
            enable_thinking=False,
            thinking_budget=512,
        )

        self.assertEqual(result, ("generated response",))
        mock_execute.assert_called_once_with(
            mlx_model=mocked_model,
            prompt="Hello",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            seed=42,
            draft_model=None,
            enable_thinking=False,
            thinking_budget=512,
        )

    @patch.object(import_node_module("generate_nodes"), "execute_text_generation")
    def test_mlx_lm_generate_happy_path_without_chat_template(self, mock_execute):
        node = self.MLXLMGenerateText()
        mocked_model = self.get_mocked_model()
        mocked_model.family = "mlx-lm"

        mock_execute.return_value = "generated response no template"

        result = node.generate(
            mlx_model=mocked_model,
            prompt="Hello raw",
            max_tokens=50,
            temperature=1.0,
            top_p=0.5,
            seed=123,
            draft_model=None,
            enable_thinking=False,
            thinking_budget=512,
        )

        self.assertEqual(result, ("generated response no template",))
        mock_execute.assert_called_once_with(
            mlx_model=mocked_model,
            prompt="Hello raw",
            max_tokens=50,
            temperature=1.0,
            top_p=0.5,
            seed=123,
            draft_model=None,
            enable_thinking=False,
            thinking_budget=512,
        )

    # --- MLXVLMDescribeImage Tests ---

    def test_mlx_vlm_run_unknown_model_family_raises_value_error(self):
        node = self.MLXVLMDescribeImage()
        mocked_model = self.get_mocked_model()
        mocked_model.family = "unknown-family"

        with self.assertRaises(ValueError) as context:
            node.run(
                mlx_model=mocked_model,
                prompt="test prompt",
                max_tokens=100,
                temperature=0.7,
                seed=42,
                enable_thinking=False,
                thinking_budget=100,
            )

        self.assertEqual(
            str(context.exception),
            "Expected model family 'mlx-vlm' but found 'unknown-family'. Please ensure you are passing a Vision-Language Model loaded via 'MLX Load Model', not a standard text or SAM model.",
        )

    @patch.object(import_node_module("generate_nodes"), "execute_image_description")
    def test_mlx_vlm_run_happy_path_no_draft_model(self, mock_execute):
        node = self.MLXVLMDescribeImage()
        mocked_model = self.get_mocked_model()
        mocked_model.family = "mlx-vlm"

        mock_execute.return_value = "image described"

        mock_image = MagicMock()

        result = node.run(
            mlx_model=mocked_model,
            prompt="Describe this",
            max_tokens=256,
            temperature=0.8,
            seed=99,
            enable_thinking=True,
            thinking_budget=512,
            image=mock_image,
            audio_path="fake/path.mp3",
            draft_model=None,
        )

        self.assertEqual(result, ("image described",))
        mock_execute.assert_called_once_with(
            mlx_model=mocked_model,
            prompt="Describe this",
            max_tokens=256,
            temperature=0.8,
            seed=99,
            enable_thinking=True,
            thinking_budget=512,
            image=mock_image,
            audio_path="fake/path.mp3",
            draft_model=None,
            draft_kind="dflash",
        )

    @patch.object(import_node_module("generate_nodes"), "execute_image_description")
    def test_mlx_vlm_run_happy_path_with_draft_model(self, mock_execute):
        node = self.MLXVLMDescribeImage()
        mocked_model = self.get_mocked_model()
        mocked_model.family = "mlx-vlm"

        mock_execute.return_value = "fast image described"

        result = node.run(
            mlx_model=mocked_model,
            prompt="Draft this",
            max_tokens=128,
            temperature=0.5,
            seed=1,
            enable_thinking=False,
            thinking_budget=0,
            image=None,
            audio_path="",
            draft_model="mock_draft_model",
            draft_kind="eagle3",
        )

        self.assertEqual(result, ("fast image described",))
        mock_execute.assert_called_once_with(
            mlx_model=mocked_model,
            prompt="Draft this",
            max_tokens=128,
            temperature=0.5,
            seed=1,
            enable_thinking=False,
            thinking_budget=0,
            image=None,
            audio_path="",
            draft_model="mock_draft_model",
            draft_kind="eagle3",
        )

    def test_batch_vlm_describe_image(self):
        MLXBatchVLMDescribeImage = self.generate_nodes.MLXBatchVLMDescribeImage

        node = MLXBatchVLMDescribeImage()

        self.assertEqual(MLXBatchVLMDescribeImage.OUTPUT_IS_LIST, (True, False))

        mock_loaded_model = self.get_mocked_model()
        mock_loaded_model.family = "mlx-vlm"

        with patch("comfyui_mlx_universal.nodes.generate_nodes.torch") as mock_torch:
            image_tensor = MagicMock()

            with patch(
                "comfyui_mlx_universal.runtime.generate_processing.execute_batch_image_description"
            ) as mock_execute:
                mock_execute.return_value = ["Response 1", "Response 2"]

                result = node.run(
                    mlx_model=mock_loaded_model,
                    image=image_tensor,
                    prompt="Test prompt",
                    max_tokens=10,
                    temperature=0.7,
                    seed=42,
                    enable_thinking=False,
                    thinking_budget=512,
                )

                self.assertIsInstance(result, tuple)
                self.assertEqual(len(result), 2)
                self.assertEqual(result[0], ["Response 1", "Response 2"])
                self.assertEqual(result[1], "Response 1\nResponse 2")

                mock_execute.assert_called_once_with(
                    mlx_model=mock_loaded_model,
                    prompt="Test prompt",
                    max_tokens=10,
                    temperature=0.7,
                    seed=42,
                    enable_thinking=False,
                    thinking_budget=512,
                    image=image_tensor,
                    draft_model=None,
                    draft_kind="dflash",
                )

    def test_batch_vlm_describe_image_wrong_family(self):
        MLXBatchVLMDescribeImage = self.generate_nodes.MLXBatchVLMDescribeImage
        node = MLXBatchVLMDescribeImage()

        mock_loaded_model = self.get_mocked_model()
        mock_loaded_model.family = "mlx-lm"

        with self.assertRaisesRegex(ValueError, "Expected model family 'mlx-vlm'"):
            node.run(
                mlx_model=mock_loaded_model,
                image=MagicMock(),
                prompt="Test",
                max_tokens=10,
                temperature=0.7,
                seed=42,
                enable_thinking=False,
                thinking_budget=512,
            )


if __name__ == "__main__":
    unittest.main()
