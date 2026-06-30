import unittest
from unittest.mock import MagicMock

from comfyui_mlx_universal.runtime.diffusion_processing import (
    encode_clip_text,
    generate_image,
)


class TestRuntimeDiffusion(unittest.TestCase):
    def test_encode_clip_text(self):
        # We just want to mock the components of the conditioning dict
        mock_tokenizer = MagicMock()
        mock_tokenizer.tokenize.return_value = [1, 2, 3]

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


if __name__ == "__main__":
    unittest.main()
