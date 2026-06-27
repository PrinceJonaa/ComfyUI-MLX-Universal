import sys
import unittest
from unittest.mock import MagicMock, patch

from tests.test_helper import import_node_module

class TestDiffusionNodes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.diffusion_nodes = import_node_module("diffusion_nodes")
        cls.MLXDecoder = cls.diffusion_nodes.MLXDecoder
        cls.MLXSampler = cls.diffusion_nodes.MLXSampler
        cls.MLXLoadFlux = cls.diffusion_nodes.MLXLoadFlux
        cls.MLXClipTextEncoder = cls.diffusion_nodes.MLXClipTextEncoder
        cls.MLXEncoder = cls.diffusion_nodes.MLXEncoder

        # Get the bridge module mock
        cls.bridge = sys.modules["comfyui_mlx_universal.runtime.bridge"]

    def setUp(self):
        # Reset bridge mocks
        self.bridge.latent_to_mlx = MagicMock()
        self.bridge.mlx_to_torch = MagicMock()
        self.bridge.torch_to_mlx = MagicMock()
        self.bridge.mlx_to_latent = MagicMock()

    def test_mlx_decoder_happy_path(self):
        node = self.MLXDecoder()
        
        # Configure mocks to support math operators
        mock_vae = MagicMock()
        mock_decoded = MagicMock()
        mock_vae.return_value = mock_decoded
        
        self.bridge.latent_to_mlx.return_value = "mock_latent"
        self.bridge.mlx_to_torch.return_value = "torch_tensor"

        result = node.decode({"samples": "mock_samples"}, mock_vae)

        self.bridge.latent_to_mlx.assert_called_once_with({"samples": "mock_samples"})
        mock_vae.assert_called_once_with("mock_latent")
        self.bridge.mlx_to_torch.assert_called_once()
        self.assertEqual(result, ("torch_tensor",))

    def test_mlx_encoder_happy_path(self):
        node = self.MLXEncoder()

        # Configure mocks to support unpacking (mean, _) from split()
        mock_model = MagicMock()
        mock_hidden = MagicMock()
        mock_model.encoder.return_value = mock_hidden
        mock_hidden.split.return_value = (MagicMock(), MagicMock())
        
        self.bridge.torch_to_mlx.return_value = MagicMock()
        self.bridge.mlx_to_latent.return_value = "mock_latent_dict"

        result = node.encode("image_tensor", mock_model)

        self.bridge.torch_to_mlx.assert_called_once_with("image_tensor")
        mock_model.encoder.assert_called_once()
        self.bridge.mlx_to_latent.assert_called_once()
        self.assertEqual(result, ("mock_latent_dict",))

    def test_mlx_sampler_validation_failures(self):
        node = self.MLXSampler()

        # Validation: missing conditioning keys
        with self.assertRaises(ValueError) as context:
            node.generate_image(
                mlx_model=MagicMock(),
                seed=42,
                steps=4,
                cfg=0.0,
                mlx_positive_conditioning={},
                latent_image={"samples": MagicMock()},
                denoise=1.0
            )
        self.assertIn("Expected a valid MLX conditioning dictionary", str(context.exception))

        # Validation: missing latent samples
        with self.assertRaises(ValueError) as context:
            node.generate_image(
                mlx_model=MagicMock(),
                seed=42,
                steps=4,
                cfg=0.0,
                mlx_positive_conditioning={"conditioning": "c", "pooled_conditioning": "pc"},
                latent_image={},
                denoise=1.0
            )
        self.assertIn("Expected a valid ComfyUI latent dictionary", str(context.exception))

    def test_mlx_sampler_happy_path(self):
        node = self.MLXSampler()

        # Setup mock inputs
        mock_model = MagicMock()
        mock_latents = MagicMock()
        mock_model.denoise_latents.return_value = (mock_latents, 0.5)
        
        self.bridge.mlx_to_latent.return_value = "output_latent_dict"
        self.bridge.latent_to_mlx.return_value = "mock_input_latents_mlx"

        mock_latent_samples = MagicMock()
        mock_latent_samples.shape = (1, 4, 64, 64)

        result = node.generate_image(
            mlx_model=mock_model,
            seed=42,
            steps=4,
            cfg=0.0,
            mlx_positive_conditioning={"conditioning": "c", "pooled_conditioning": "pc"},
            latent_image={"samples": mock_latent_samples},
            denoise=0.8
        )

        self.bridge.latent_to_mlx.assert_called_once_with({"samples": mock_latent_samples})
        mock_model.denoise_latents.assert_called_once()
        mock_latents.astype.assert_called_once_with(mock_model.activation_dtype)
        self.assertEqual(result, ("output_latent_dict",))

    @patch("comfyui_mlx_universal.runtime.model_loader.load_flux_pipeline")
    def test_mlx_load_flux_happy_path(self, mock_load_pipeline):
        mock_model = MagicMock()
        mock_load_pipeline.return_value = mock_model

        node = self.MLXLoadFlux()
        result = node.load_flux_model("argmaxinc/mlx-FLUX.1-schnell")

        mock_load_pipeline.assert_called_once_with("argmaxinc/mlx-FLUX.1-schnell")
        self.assertEqual(result[0], mock_model)
        self.assertEqual(result[1], mock_model.decoder)

    def test_mlx_clip_text_encoder_happy_path(self):
        node = self.MLXClipTextEncoder()

        # Setup mock conditioning
        mock_conditioning = {
            "model_name": "argmaxinc/mlx-FLUX.1-schnell",
            "clip_l_model": MagicMock(),
            "clip_l_tokenizer": MagicMock(),
            "t5_model": MagicMock(),
            "t5_tokenizer": MagicMock(),
        }

        # Mock tokenizer behaviors
        mock_conditioning["clip_l_tokenizer"].tokenize.return_value = [1, 2, 3]
        mock_conditioning["clip_l_tokenizer"].pad_with_eos = True
        mock_conditioning["clip_l_tokenizer"].pad_to_max_length = False

        mock_conditioning["t5_tokenizer"].tokenize.return_value = [4, 5]
        mock_conditioning["t5_tokenizer"].pad_with_eos = True
        mock_conditioning["t5_tokenizer"].pad_to_max_length = False

        # Mock model forward passes
        mock_clip_out = MagicMock()
        mock_clip_out.pooled_output = "mock_pooled"
        mock_conditioning["clip_l_model"].return_value = mock_clip_out
        mock_conditioning["t5_model"].return_value = "mock_t5_out"

        result = node.encode(mock_conditioning, "A photo of a dog")

        self.assertEqual(result[0]["pooled_conditioning"], "mock_pooled")
        self.assertEqual(result[0]["conditioning"], "mock_t5_out")


if __name__ == "__main__":
    unittest.main()
