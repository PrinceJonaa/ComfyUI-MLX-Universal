import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Create mock objects for the dependencies
mock_mlx = MagicMock()
sys.modules['mlx'] = mock_mlx

# Note: mx is what generate_nodes.py imports directly via `import mlx.core as mx`
# But if it wasn't mocked properly, let's inject it into the exec namespace directly
# We can just provide `mx` inside the dummy namespace.
mock_mx = MagicMock()

mock_mlx_lm = MagicMock()
sys.modules['mlx_lm'] = mock_mlx_lm

mock_mlx_lm_sample_utils = MagicMock()
sys.modules['mlx_lm.sample_utils'] = mock_mlx_lm_sample_utils

mock_mlx_vlm = MagicMock()
sys.modules['mlx_vlm'] = mock_mlx_vlm

mock_mlx_vlm_prompt_utils = MagicMock()
sys.modules['mlx_vlm.prompt_utils'] = mock_mlx_vlm_prompt_utils

mock_mlx_vlm_speculative = MagicMock()
sys.modules['mlx_vlm.speculative'] = mock_mlx_vlm_speculative

mock_mlx_vlm_speculative_drafters = MagicMock()
sys.modules['mlx_vlm.speculative.drafters'] = mock_mlx_vlm_speculative_drafters

class TestGenerateNodes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(root_dir, 'nodes', 'generate_nodes.py')

        with open(file_path, 'r') as f:
            code = f.read()

        # Strip out relative imports and inject mocks
        lines = code.split('\n')
        new_lines = []
        for line in lines:
            if line.startswith('from ..runtime'):
                continue
            if line.startswith('import mlx.core as mx'):
                # We provide `mx` dynamically
                continue
            new_lines.append(line)

        # Add dummy LoadedMLXModel and other dependencies
        dummy_classes = """
class LoadedMLXModel:
    def __init__(self):
        self.family = ""
        self.model = None
        self.processor = None

def tensor_to_pil(*args, **kwargs):
    return ["mocked_pil_image"] if args[0] is not None else []

def get_or_load_draft_model(key, loader):
    return ("mock_draft_model", "mock_draft_kind")

def make_key(*args, **kwargs):
    return "mock_key"
"""
        new_code = dummy_classes + '\n'.join(new_lines)

        namespace = {}
        # Make os and mx available in namespace
        namespace['os'] = os
        namespace['mx'] = mock_mx
        exec(new_code, namespace)

        cls.MLXLMGenerateText = namespace['MLXLMGenerateText']
        cls.MLXVLMDescribeImage = namespace['MLXVLMDescribeImage']
        cls.LoadedMLXModel = namespace['LoadedMLXModel']

    def setUp(self):
        # Reset mocks before each test
        mock_mx.random.seed.reset_mock()
        mock_mlx_lm.generate.reset_mock()
        if hasattr(mock_mlx_lm, "sample_utils"):
            mock_mlx_lm.sample_utils.make_sampler.reset_mock()
        mock_mlx_vlm.generate.reset_mock()
        mock_mlx_vlm_prompt_utils.apply_chat_template.reset_mock()
        mock_mlx_vlm_speculative_drafters.load_drafter.reset_mock()

    def get_mocked_model(self):
        model = self.LoadedMLXModel()
        model.model = MagicMock()
        model.processor = MagicMock()
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
                seed=42
            )

        self.assertEqual(str(context.exception), "Expected model family 'mlx-lm'. Found 'unknown-family'. Please load an mlx-lm model.")

    @patch('mlx_lm.sample_utils.make_sampler')
    def test_mlx_lm_generate_happy_path_with_chat_template(self, mock_make_sampler):
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
            seed=42
        )

        self.assertEqual(result, ("generated response",))
        mock_mx.random.seed.assert_called_once_with(42)
        mocked_model.processor.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "Hello"}], tokenize=False, add_generation_prompt=True
        )
        mock_make_sampler.assert_called_once_with(temp=0.7, top_p=0.9)
        mock_mlx_lm.generate.assert_called_once_with(
            mocked_model.model,
            mocked_model.processor,
            prompt="formatted_prompt",
            sampler="mocked_sampler",
            max_tokens=100,
            verbose=False
        )

    @patch('mlx_lm.sample_utils.make_sampler')
    def test_mlx_lm_generate_happy_path_without_chat_template(self, mock_make_sampler):
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
            seed=123
        )

        self.assertEqual(result, ("generated response no template",))
        mock_mx.random.seed.assert_called_once_with(123)
        mock_make_sampler.assert_called_once_with(temp=1.0, top_p=0.5)
        mock_mlx_lm.generate.assert_called_once_with(
            mocked_model.model,
            mocked_model.processor,
            prompt="Hello raw",
            sampler="mocked_sampler_2",
            max_tokens=50,
            verbose=False
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
                thinking_budget=100
            )

        self.assertEqual(str(context.exception), "Expected model family 'mlx-vlm'. Found 'unknown-family'. Please load an mlx-vlm model.")

    @patch('os.path.exists', return_value=True)
    def test_mlx_vlm_run_happy_path_no_draft_model(self, mock_os_exists):
        node = self.MLXVLMDescribeImage()
        mocked_model = self.get_mocked_model()
        mocked_model.family = "mlx-vlm"

        mock_mlx_vlm_prompt_utils.apply_chat_template.return_value = "vlm_formatted_prompt"
        mock_mlx_vlm.generate.return_value = "image described"

        result = node.run(
            mlx_model=mocked_model,
            prompt="Describe this",
            max_tokens=256,
            temperature=0.8,
            seed=99,
            enable_thinking=True,
            thinking_budget=512,
            image="mock_tensor",
            audio_path="fake/path.mp3",
            draft_model_path=""
        )

        self.assertEqual(result, ("image described",))
        mock_mx.random.seed.assert_called_once_with(99)
        mock_mlx_vlm_prompt_utils.apply_chat_template.assert_called_once_with(
            mocked_model.processor,
            mocked_model.model.config,
            "Describe this",
            num_images=1, # Since tensor_to_pil returns 1 mock item
            num_audios=1
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
            thinking_budget=512
        )

    @patch('os.path.exists', return_value=False)
    def test_mlx_vlm_run_happy_path_with_draft_model(self, mock_os_exists):
        node = self.MLXVLMDescribeImage()
        mocked_model = self.get_mocked_model()
        mocked_model.family = "mlx-vlm"

        mock_mlx_vlm_prompt_utils.apply_chat_template.return_value = "vlm_formatted_prompt_draft"
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
            draft_kind="eagle3"
        )

        self.assertEqual(result, ("fast image described",))
        mock_mx.random.seed.assert_called_once_with(1)
        mock_mlx_vlm_prompt_utils.apply_chat_template.assert_called_once_with(
            mocked_model.processor,
            mocked_model.model.config,
            "Draft this",
            num_images=0,
            num_audios=0
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
            draft_kind="mock_draft_kind"
        )

if __name__ == '__main__':
    unittest.main()
