import os
import sys
import unittest
from unittest.mock import MagicMock

from tests.test_helper import import_node_module


@unittest.skipIf(
    os.environ.get("REAL_MLX_TESTS") == "1", "Skipping mock tests with real MLX"
)
class TestLoaderNodes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader_nodes = import_node_module("loader_nodes")
        cls.MLXModelLoaderUnified = cls.loader_nodes.MLXModelLoaderUnified
        cls.MLXApplyLoRA = cls.loader_nodes.MLXApplyLoRA

        # Get the real LoadedMLXModel dataclass
        runtime_data_types = sys.modules["comfyui_mlx_universal.runtime.data_types"]
        cls.LoadedMLXModel = runtime_data_types.LoadedMLXModel

    def setUp(self):
        # Configure manual clean mock for load_unified_mlx_model in the loader_nodes module
        self.loader_nodes.load_unified_mlx_model = MagicMock()

    def test_mlx_model_loader_unified_calls_runtime_loader(self):
        self.loader_nodes.load_unified_mlx_model.return_value = "mock_loaded_model"

        node = self.MLXModelLoaderUnified()
        result = node.load_model(
            model_path="mlx-community/Qwen3.5-4B",
            model_type="mlx-lm",
            trust_remote_code=True,
            quantize_activations=True,
            adapter_path="lora/path",
        )

        self.loader_nodes.load_unified_mlx_model.assert_called_once_with(
            model_path="mlx-community/Qwen3.5-4B",
            model_type="mlx-lm",
            trust_remote_code=True,
            quantize_activations=True,
            adapter_path="lora/path",
        )
        self.assertEqual(result, ("mock_loaded_model",))

    def test_mlx_apply_lora_empty_adapter_returns_model_untouched(self):
        node = self.MLXApplyLoRA()
        mock_model = MagicMock(spec=self.LoadedMLXModel)

        result = node.apply_lora(mock_model, "")

        self.loader_nodes.load_unified_mlx_model.assert_not_called()
        self.assertEqual(result, (mock_model,))

    def test_mlx_apply_lora_dynamic_fusion_for_mlx_lm(self):
        node = self.MLXApplyLoRA()
        mock_internal_model = MagicMock()
        mock_processor = MagicMock()

        mock_model = self.LoadedMLXModel(
            family="mlx-lm",
            model_path="original/path",
            model_type="mlx-lm",
            trust_remote_code=False,
            quantize_activations=True,
            model=mock_internal_model,
            processor=mock_processor,
        )

        from unittest.mock import patch

        with patch("copy.deepcopy") as mock_deepcopy:

            mock_copied_model = MagicMock()
            mock_deepcopy.return_value = mock_copied_model
            mock_patched_model = MagicMock()

            import mlx_lm.utils
            mock_load_adapters = MagicMock(return_value=mock_patched_model)
            mlx_lm.utils.load_adapters = mock_load_adapters

            result = node.apply_lora(mock_model, "lora/adapter")

            mock_deepcopy.assert_called_once_with(mock_internal_model)
            mock_load_adapters.assert_called_once_with(mock_copied_model, "lora/adapter")

            self.loader_nodes.load_unified_mlx_model.assert_not_called()

            self.assertIsInstance(result[0], self.LoadedMLXModel)
            self.assertEqual(result[0].model, mock_patched_model)
            self.assertEqual(result[0].processor, mock_processor)
            self.assertEqual(result[0].model_path, "original/path")

    def test_mlx_apply_lora_fallback_fusion_for_non_mlx_lm(self):
        self.loader_nodes.load_unified_mlx_model.return_value = "mock_fused_model"
        node = self.MLXApplyLoRA()

        mock_model = self.LoadedMLXModel(
            family="mlx-vlm",
            model_path="original/path",
            model_type="mlx-vlm",
            trust_remote_code=False,
            quantize_activations=True,
            model=MagicMock(),
            processor=MagicMock(),
        )

        result = node.apply_lora(mock_model, "lora/adapter")

        self.loader_nodes.load_unified_mlx_model.assert_called_once_with(
            model_path="original/path",
            model_type="mlx-vlm",
            trust_remote_code=False,
            quantize_activations=True,
            adapter_path="lora/adapter",
        )
        self.assertEqual(result, ("mock_fused_model",))

    def test_mlx_draft_model_loader_calls_runtime_loader(self):
        from unittest.mock import patch

        with patch.object(self.loader_nodes, "load_draft_model") as mock_load:
            mock_load.return_value = "mock_draft_model"
            node = self.loader_nodes.MLXDraftModelLoader()

            result = node.load_model(
                model_path="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                model_family="mlx-lm",
            )

            mock_load.assert_called_once_with(
                "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                "mlx-lm",
            )
            self.assertEqual(result, ("mock_draft_model",))


if __name__ == "__main__":
    unittest.main()
