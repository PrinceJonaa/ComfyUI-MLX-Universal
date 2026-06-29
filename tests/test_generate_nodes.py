import sys
import os
import unittest
from unittest.mock import MagicMock, patch

from tests.test_helper import import_node_module

# Create mock objects for the dependencies
mock_mlx_lm = sys.modules["mlx_lm"]
mock_mlx_lm_sample_utils = sys.modules["mlx_lm.sample_utils"]
mock_mlx_vlm = sys.modules["mlx_vlm"]
mock_mlx_vlm_prompt_utils = sys.modules["mlx_vlm.prompt_utils"]
mock_mlx_vlm_speculative = sys.modules["mlx_vlm.speculative"]
mock_mlx_vlm_speculative_drafters = sys.modules["mlx_vlm.speculative.drafters"]


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
        # Reset mocks before each test
        self.generate_nodes.mx.random.seed.reset_mock()
        mock_mlx_lm.generate.reset_mock()
        mock_mlx_lm_sample_utils.make_sampler.reset_mock()
        mock_mlx_vlm.generate.reset_mock()
        mock_mlx_vlm_prompt_utils.apply_chat_template.reset_mock()
        mock_mlx_vlm_speculative_drafters.load_drafter.reset_mock()
        
        # Setup clean mock for tensor_to_pil in the generate_nodes module
        self.generate_nodes.tensor_to_pil = MagicMock(return_value=["mocked_pil_image"])

    def get_mocked_model(self):
        model = self.LoadedMLXModel(
            family="mlx-lm",
            model_path="mock_path",
            model_type="mlx-lm",
            trust_remote_code=False,
            quantize_activations=False,
            model=MagicMock(),
            processor=MagicMock()
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
                draft_model_path="",
                enable_thinking=False,
                thinking_budget=512,
            )

        self.assertEqual(
            str(context.exception),
            "Expected model family 'mlx-lm' but found 'unknown-family'. Please ensure you are passing a text model loaded via 'MLX Load Model', not a Vision, Audio, or SAM model.",
        )

    @patch("comfyui_mlx_universal.runtime.model_loader.load_draft_model")
    @patch("mlx_lm.sample_utils.make_sampler")
    def test_mlx_lm_generate_happy_path_with_chat_template(self, mock_make_sampler, mock_load_draft):
        mock_make_sampler.return_value = "mocked_sampler"
        node = self.MLXLMGenerateText()
        mocked_model = self.get_mocked_model()
        mocked_model.family = "mlx-lm"
        mocked_model.processor.chat_template = "template"
        mocked_model.processor.apply_chat_template.return_value = "formatted_prompt"

        mock_mlx_lm.generate.return_value = "generated response"

        result = node.generate(
            mlx_model=mocked_model,
            prompt="Hello",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            seed=42,
            draft_model_path="",
            enable_thinking=False,
            thinking_budget=512,
        )

        self.assertEqual(result, ("generated response",))
        self.generate_nodes.mx.random.seed.assert_called_once_with(42)
        mocked_model.processor.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "Hello"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        mock_make_sampler.assert_called_once_with(temp=0.7, top_p=0.9)
        mock_mlx_lm.generate.assert_called_once_with(
            mocked_model.model,
            mocked_model.processor,
            prompt="formatted_prompt",
            sampler="mocked_sampler",
            max_tokens=100,
            verbose=False,
            enable_thinking=False,
            thinking_budget=512,
        )

    @patch("comfyui_mlx_universal.runtime.model_loader.load_draft_model")
    @patch("mlx_lm.sample_utils.make_sampler")
    def test_mlx_lm_generate_happy_path_without_chat_template(self, mock_make_sampler, mock_load_draft):
        mock_make_sampler.return_value = "mocked_sampler_2"
        node = self.MLXLMGenerateText()
        mocked_model = self.get_mocked_model()
        mocked_model.family = "mlx-lm"
        # Simulate missing chat_template
        del mocked_model.processor.chat_template

        mock_mlx_lm.generate.return_value = "generated response no template"

        result = node.generate(
            mlx_model=mocked_model,
            prompt="Hello raw",
            max_tokens=50,
            temperature=1.0,
            top_p=0.5,
            seed=123,
            draft_model_path="",
            enable_thinking=False,
            thinking_budget=512,
        )

        self.assertEqual(result, ("generated response no template",))
        self.generate_nodes.mx.random.seed.assert_called_once_with(123)
        mock_make_sampler.assert_called_once_with(temp=1.0, top_p=0.5)
        mock_mlx_lm.generate.assert_called_once_with(
            mocked_model.model,
            mocked_model.processor,
            prompt="Hello raw",
            sampler="mocked_sampler_2",
            max_tokens=50,
            verbose=False,
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

    @patch("comfyui_mlx_universal.runtime.model_loader.load_draft_model")
    @patch("os.path.exists", return_value=True)
    def test_mlx_vlm_run_happy_path_no_draft_model(self, mock_os_exists, mock_load_draft):
        node = self.MLXVLMDescribeImage()
        mocked_model = self.get_mocked_model()
        mocked_model.family = "mlx-vlm"

        mock_mlx_vlm_prompt_utils.apply_chat_template.return_value = (
            "vlm_formatted_prompt"
        )
        mock_mlx_vlm.generate.return_value = "image described"

        # Pass a mock image tensor with ndim property
        mock_image = MagicMock()
        mock_image.ndim = 3

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
            draft_model_path="",
        )

        self.assertEqual(result, ("image described",))
        self.generate_nodes.mx.random.seed.assert_called_once_with(99)
        mock_mlx_vlm_prompt_utils.apply_chat_template.assert_called_once_with(
            mocked_model.processor,
            mocked_model.model.config,
            "Describe this",
            num_images=1,
            num_audios=1,
        )
        mock_mlx_vlm.generate.assert_called_once_with(
            mocked_model.model,
            mocked_model.processor,
            "vlm_formatted_prompt",
            image=["mocked_pil_image"],
            audio=["fake/path.mp3"],
            temp=0.8,
            max_tokens=256,
            verbose=False,
            enable_thinking=True,
            thinking_budget=512,
        )

    @patch("comfyui_mlx_universal.runtime.model_loader.load_draft_model")
    @patch("os.path.exists", return_value=False)
    def test_mlx_vlm_run_happy_path_with_draft_model(self, mock_os_exists, mock_load_draft):
        node = self.MLXVLMDescribeImage()
        mocked_model = self.get_mocked_model()
        mocked_model.family = "mlx-vlm"
        mock_load_draft.return_value = "mock_draft_model"

        # Ensure tensor_to_pil returns empty when no image
        self.generate_nodes.tensor_to_pil.return_value = []

        mock_mlx_vlm_prompt_utils.apply_chat_template.return_value = (
            "vlm_formatted_prompt_draft"
        )
        mock_mlx_vlm.generate.return_value = "fast image described"

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
            draft_model_path="path/to/draft",
            draft_kind="eagle3",
        )

        self.assertEqual(result, ("fast image described",))
        self.generate_nodes.mx.random.seed.assert_called_once_with(1)
        mock_mlx_vlm_prompt_utils.apply_chat_template.assert_called_once_with(
            mocked_model.processor,
            mocked_model.model.config,
            "Draft this",
            num_images=0,
            num_audios=0,
        )
        mock_mlx_vlm.generate.assert_called_once_with(
            mocked_model.model,
            mocked_model.processor,
            "vlm_formatted_prompt_draft",
            image=None,
            audio=None,
            temp=0.5,
            max_tokens=128,
            verbose=False,
            enable_thinking=False,
            thinking_budget=0,
            draft_model="mock_draft_model",
            draft_kind="eagle3",
        )


if __name__ == "__main__":
    unittest.main()
