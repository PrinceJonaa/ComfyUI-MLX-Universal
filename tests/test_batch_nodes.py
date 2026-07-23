import os
import sys
import unittest
from unittest.mock import ANY, MagicMock, patch

from tests.test_helper import import_node_module


@unittest.skipIf(
    os.environ.get("REAL_MLX_TESTS") == "1", "Skipping mock tests with real MLX"
)
class TestBatchNodes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_nodes = import_node_module("batch_nodes")
        cls.MLXBatchVLMDescribeImage = cls.batch_nodes.MLXBatchVLMDescribeImage
        cls.LoadedMLXModel = sys.modules[
            "comfyui_mlx_universal.runtime.data_types"
        ].LoadedMLXModel

    def test_batch_vlm_node_delegates_to_runtime(self):
        model = self.LoadedMLXModel(
            family="mlx-vlm",
            model_path="mock-path",
            model_type="mlx-vlm",
            trust_remote_code=False,
            quantize_activations=False,
            model=MagicMock(),
            processor=MagicMock(),
        )
        node = self.MLXBatchVLMDescribeImage()

        with patch.object(
            self.batch_nodes, "execute_batch_image_description", return_value="done"
        ) as execute:
            result = node.run(
                mlx_model=model,
                prompt="Describe these images",
                max_tokens=128,
                temperature=0.5,
                seed=7,
                enable_thinking=False,
                thinking_budget=0,
                image=MagicMock(),
            )

        self.assertEqual(result, ("done",))
        execute.assert_called_once_with(
            mlx_model=model,
            prompt="Describe these images",
            max_tokens=128,
            temperature=0.5,
            seed=7,
            enable_thinking=False,
            thinking_budget=0,
            image=ANY,
            audio_path="",
            draft_model=None,
            draft_kind="dflash",
        )

    def test_batch_lm_generate_text_delegates_to_runtime(self):
        model = self.LoadedMLXModel(
            family="mlx-lm",
            model_path="mock-path",
            model_type="mlx-lm",
            trust_remote_code=False,
            quantize_activations=False,
            model=MagicMock(),
            processor=MagicMock(),
        )
        MLXBatchLMGenerateText = self.batch_nodes.MLXBatchLMGenerateText
        node = MLXBatchLMGenerateText()

        with patch.object(
            self.batch_nodes, "execute_batch_text_generation", return_value="done"
        ) as execute:
            result = node.generate(
                mlx_model=model,
                prompts="prompt1\n---\nprompt2",
                delimiter="\n---\n",
                max_tokens=128,
                temperature=0.5,
                top_p=0.9,
                seed=7,
                draft_model=None,
                enable_thinking=False,
                thinking_budget=0,
            )

        self.assertEqual(result, ("done",))
        execute.assert_called_once_with(
            mlx_model=model,
            prompts="prompt1\n---\nprompt2",
            delimiter="\n---\n",
            max_tokens=128,
            temperature=0.5,
            top_p=0.9,
            seed=7,
            draft_model=None,
            enable_thinking=False,
            thinking_budget=0,
        )
