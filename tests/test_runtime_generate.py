import unittest
from unittest.mock import MagicMock, patch

from comfyui_mlx_universal.runtime.data_types import LoadedMLXModel
from comfyui_mlx_universal.runtime.generate_processing import (
    execute_image_description,
    execute_text_generation,
)


class TestRuntimeGenerate(unittest.TestCase):
    def get_mocked_model(self):
        model = LoadedMLXModel(
            family="mlx-lm",
            model_path="mock_path",
            model_type="mlx-lm",
            trust_remote_code=False,
            quantize_activations=False,
            model=MagicMock(),
            processor=MagicMock(),
        )
        return model

    # --- execute_text_generation Tests ---

    @patch("comfyui_mlx_universal.runtime.generate_processing.mx")
    @patch("mlx_lm.generate")
    @patch("mlx_lm.sample_utils.make_sampler")
    def test_execute_text_generation_with_chat_template(
        self, mock_make_sampler, mock_generate, mock_mx
    ):
        mock_make_sampler.return_value = "mocked_sampler"
        mocked_model = self.get_mocked_model()
        mocked_model.processor.chat_template = "template"
        mocked_model.processor.apply_chat_template.return_value = "formatted_prompt"

        mock_generate.return_value = "generated response"

        result = execute_text_generation(
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

        self.assertEqual(result, "generated response")
        mock_mx.random.seed.assert_called_once_with(42)
        mocked_model.processor.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "Hello"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        mock_make_sampler.assert_called_once_with(temp=0.7, top_p=0.9)
        mock_generate.assert_called_once_with(
            mocked_model.model,
            mocked_model.processor,
            prompt="formatted_prompt",
            sampler="mocked_sampler",
            max_tokens=100,
            verbose=False,
        )

    @patch("comfyui_mlx_universal.runtime.generate_processing.mx")
    @patch("mlx_lm.generate")
    @patch("mlx_lm.sample_utils.make_sampler")
    def test_execute_text_generation_without_chat_template(
        self, mock_make_sampler, mock_generate, mock_mx
    ):
        mock_make_sampler.return_value = "mocked_sampler_2"
        mocked_model = self.get_mocked_model()
        # Simulate missing chat_template
        del mocked_model.processor.chat_template

        mock_generate.return_value = "generated response no template"

        result = execute_text_generation(
            mlx_model=mocked_model,
            prompt="Hello raw",
            max_tokens=50,
            temperature=1.0,
            top_p=0.5,
            seed=123,
            draft_model="mock_draft_model",
            enable_thinking=True,
            thinking_budget=1024,
        )

        self.assertEqual(result, "generated response no template")
        mock_mx.random.seed.assert_called_once_with(123)
        mock_make_sampler.assert_called_once_with(temp=1.0, top_p=0.5)
        mock_generate.assert_called_once_with(
            mocked_model.model,
            mocked_model.processor,
            prompt="Hello raw",
            sampler="mocked_sampler_2",
            max_tokens=50,
            verbose=False,
            draft_model="mock_draft_model",
        )

    # --- execute_image_description Tests ---

    @patch("comfyui_mlx_universal.runtime.generate_processing.mx")
    @patch("mlx_vlm.generate")
    @patch("mlx_vlm.prompt_utils.apply_chat_template")
    @patch("os.path.exists", return_value=True)
    @patch("comfyui_mlx_universal.runtime.generate_processing.tensor_to_pil")
    def test_execute_image_description_no_draft_model(
        self,
        mock_tensor_to_pil,
        mock_os_exists,
        mock_apply_chat_template,
        mock_generate,
        mock_mx,
    ):
        mock_tensor_to_pil.return_value = ["mocked_pil_image"]
        mocked_model = self.get_mocked_model()

        mock_apply_chat_template.return_value = "vlm_formatted_prompt"
        mock_generate.return_value = "image described"

        mock_image = MagicMock()

        result = execute_image_description(
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

        self.assertEqual(result, "image described")
        mock_mx.random.seed.assert_called_once_with(99)
        mock_apply_chat_template.assert_called_once_with(
            mocked_model.processor,
            mocked_model.model.config,
            "Describe this",
            num_images=1,
            num_audios=1,
        )
        mock_generate.assert_called_once_with(
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

    @patch("comfyui_mlx_universal.runtime.generate_processing.mx")
    @patch("mlx_vlm.generate")
    @patch("mlx_vlm.prompt_utils.apply_chat_template")
    @patch("os.path.exists", return_value=False)
    @patch("comfyui_mlx_universal.runtime.generate_processing.tensor_to_pil")
    def test_execute_image_description_with_draft_model(
        self,
        mock_tensor_to_pil,
        mock_os_exists,
        mock_apply_chat_template,
        mock_generate,
        mock_mx,
    ):
        mock_tensor_to_pil.return_value = []
        mocked_model = self.get_mocked_model()

        mock_apply_chat_template.return_value = "vlm_formatted_prompt_draft"
        mock_generate.return_value = "fast image described"

        result = execute_image_description(
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

        self.assertEqual(result, "fast image described")
        mock_mx.random.seed.assert_called_once_with(1)
        mock_apply_chat_template.assert_called_once_with(
            mocked_model.processor,
            mocked_model.model.config,
            "Draft this",
            num_images=0,
            num_audios=0,
        )
        mock_generate.assert_called_once_with(
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
