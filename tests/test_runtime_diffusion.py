import os
import unittest
from unittest.mock import MagicMock, patch

from comfyui_mlx_universal.runtime.diffusion_processing import (
    encode_clip_text,
    generate_image,
)


@unittest.skipIf(
    os.environ.get("REAL_MLX_TESTS") == "1", "Skipping mock tests with real MLX"
)
class TestRuntimeDiffusion(unittest.TestCase):
    def test_encode_clip_text(self):
        # We just want to mock the components of the conditioning dict
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = [1, 2, 3]
        import sys

        if sys.modules["mlx.core"] is not None and isinstance(
            sys.modules["mlx.core"], MagicMock
        ):
            # Ensure the mock array returned by mx.array has a valid shape property
            sys.modules["mlx.core"].array.return_value.shape = (1, 3)

        mock_clip = MagicMock()
        mock_clip_output = MagicMock()
        mock_clip_output.pooled_output = "pooled_test"
        mock_clip.return_value = mock_clip_output

        mock_t5 = MagicMock()
        mock_t5.return_value = "t5_test"

        cond_dict = {
            "model_name": "argmaxinc/mlx-FLUX.1-schnell",
            "clip_l_model": mock_clip,
            "clip_l_tokenizer": mock_tokenizer,
            "t5_model": mock_t5,
            "t5_tokenizer": mock_tokenizer,
        }

        # Test encode_clip_text
        result = encode_clip_text(cond_dict, "a cute cat")

        # Check return format
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        self.assertIn("conditioning", result[0])
        self.assertIn("pooled_conditioning", result[0])
        self.assertEqual(result[0]["conditioning"], "t5_test")
        self.assertEqual(result[0]["pooled_conditioning"], "pooled_test")

    def test_generate_image_error_handling(self):
        # Test missing conditioning key
        mock_model = MagicMock()

        with self.assertRaisesRegex(
            ValueError, "invalid or missing conditioning input"
        ):
            generate_image(
                mlx_model=mock_model,
                seed=42,
                steps=4,
                cfg=0.0,
                mlx_positive_conditioning={"wrong_key": "val"},
                latent_image={"samples": "val"},
                denoise=1.0,
            )

        with self.assertRaisesRegex(ValueError, "invalid or missing latent input"):
            generate_image(
                mlx_model=mock_model,
                seed=42,
                steps=4,
                cfg=0.0,
                mlx_positive_conditioning={
                    "conditioning": "c",
                    "pooled_conditioning": "p",
                },
                latent_image={"wrong_key": "val"},
                denoise=1.0,
            )

    @patch("comfyui_mlx_universal.runtime.diffusion_processing.mx.eval")
    @patch("comfyui_mlx_universal.runtime.diffusion_processing.mlx_to_torch")
    @patch("comfyui_mlx_universal.runtime.diffusion_processing.latent_to_mlx")
    def test_decode_latents(self, mock_latent_to_mlx, mock_mlx_to_torch, mock_mx_eval):
        mock_latent_to_mlx.return_value = "mlx_latent_mock"
        mock_mlx_to_torch.return_value = "torch_image_mock"

        mock_vae = MagicMock()

        # We need to mock the tensor math operations too
        mock_decoded = MagicMock()
        mock_vae.return_value = mock_decoded

        import sys

        # ensure clip works on the mock
        if sys.modules["mlx.core"] is not None and isinstance(
            sys.modules["mlx.core"], MagicMock
        ):
            sys.modules["mlx.core"].clip.return_value = mock_decoded
            sys.modules["mlx.core"].float32 = "float32"
            mock_decoded.astype.return_value = mock_decoded

        from comfyui_mlx_universal.runtime.diffusion_processing import decode_latents

        result = decode_latents({"samples": "mock_samples"}, mock_vae)

        self.assertEqual(result, ("torch_image_mock",))
        mock_latent_to_mlx.assert_called_once_with({"samples": "mock_samples"})
        mock_vae.assert_called_once_with("mlx_latent_mock")
        mock_mx_eval.assert_called_once()
        mock_mlx_to_torch.assert_called_once()

    @patch("comfyui_mlx_universal.runtime.diffusion_processing.mx.eval")
    @patch("comfyui_mlx_universal.runtime.diffusion_processing.mlx_to_latent")
    @patch("comfyui_mlx_universal.runtime.diffusion_processing.torch_to_mlx")
    def test_encode_image(self, mock_torch_to_mlx, mock_mlx_to_latent, mock_mx_eval):
        mock_torch_to_mlx.return_value = MagicMock()
        mock_mlx_to_latent.return_value = "latent_image_mock"

        mock_vae = MagicMock()
        mock_hidden = MagicMock()
        mock_mean = MagicMock()
        mock_hidden.split.return_value = (mock_mean, MagicMock())
        mock_vae.encoder.return_value = mock_hidden

        mock_latents = MagicMock()
        mock_vae.latent_format.process_in.return_value = mock_latents

        from comfyui_mlx_universal.runtime.diffusion_processing import encode_image

        result = encode_image("torch_image_mock", mock_vae)

        self.assertEqual(result, ("latent_image_mock",))
        mock_torch_to_mlx.assert_called_once_with("torch_image_mock")
        mock_vae.encoder.assert_called_once()
        mock_vae.latent_format.process_in.assert_called_once_with(mock_mean)
        mock_mx_eval.assert_called_once_with(mock_latents)
        mock_mlx_to_latent.assert_called_once_with(mock_latents)

    @patch("comfyui_mlx_universal.runtime.diffusion_processing.mx.eval")
    @patch("comfyui_mlx_universal.runtime.diffusion_processing.mlx_to_latent")
    @patch("comfyui_mlx_universal.runtime.diffusion_processing.latent_to_mlx")
    def test_generate_image(self, mock_latent_to_mlx, mock_mlx_to_latent, mock_mx_eval):
        mock_latent_to_mlx.return_value = "input_latents_mock"
        mock_mlx_to_latent.return_value = "latent_image_mock"

        mock_model = MagicMock()
        mock_latents = MagicMock()
        mock_latents.astype.return_value = mock_latents
        mock_model.denoise_latents.return_value = (mock_latents, 1.0)
        mock_model.activation_dtype = "float32"

        cond_dict = {"conditioning": "cond", "pooled_conditioning": "pooled"}
        latent_dict = {"samples": MagicMock()}
        latent_dict["samples"].shape = (1, 4, 64, 64)

        from comfyui_mlx_universal.runtime.diffusion_processing import generate_image

        result = generate_image(
            mlx_model=mock_model,
            seed=42,
            steps=4,
            cfg=1.0,
            mlx_positive_conditioning=cond_dict,
            latent_image=latent_dict,
            denoise=0.5,
        )

        self.assertEqual(result, ("latent_image_mock",))
        mock_latent_to_mlx.assert_called_once_with(latent_dict)
        mock_model.denoise_latents.assert_called_once_with(
            "cond",
            "pooled",
            num_steps=4,
            cfg_weight=1.0,
            latent_size=(64, 64),
            seed=42,
            image_path=None,
            denoise=0.5,
            input_latents="input_latents_mock",
        )
        mock_mx_eval.assert_called_once_with(mock_latents)
        mock_mlx_to_latent.assert_called_once_with(mock_latents)


if __name__ == "__main__":
    unittest.main()
