import unittest
import numpy as np

# We only test real MLX behavior if the environment variable is set
import os
import sys

class TestRuntimeBridge(unittest.TestCase):
    def setUp(self):
        self.use_real_mlx = os.environ.get("REAL_MLX_TESTS") == "1"
        if not self.use_real_mlx:
            self.skipTest("Skipping real MLX bridge tests; REAL_MLX_TESTS is not set.")

    def test_mlx_to_torch_and_back(self):
        from comfyui_mlx_universal.runtime.bridge import mlx_to_torch, torch_to_mlx
        import mlx.core as mx
        import torch

        # 1. Create a dummy MLX array (e.g. latent image 1x4x64x64)
        mlx_array = mx.random.normal((1, 4, 64, 64))
        
        # 2. Convert to torch
        torch_tensor = mlx_to_torch(mlx_array)
        
        self.assertIsInstance(torch_tensor, torch.Tensor)
        self.assertEqual(torch_tensor.shape, (1, 4, 64, 64))
        self.assertEqual(torch_tensor.dtype, torch.float32)

        # 3. Convert back to MLX
        mlx_array_back = torch_to_mlx(torch_tensor)
        
        self.assertTrue(isinstance(mlx_array_back, mx.array))
        self.assertEqual(mlx_array_back.shape, (1, 4, 64, 64))

        # Check precision (they should be roughly equal)
        np.testing.assert_allclose(np.array(mlx_array), np.array(mlx_array_back), rtol=1e-5)

    def test_tensor_to_pil(self):
        from comfyui_mlx_universal.runtime.bridge import tensor_to_pil
        import torch
        from PIL import Image

        # ComfyUI Image tensor is (B, H, W, C) in [0, 1]
        dummy_comfy_img = torch.rand(1, 512, 512, 3)
        pil_imgs = tensor_to_pil(dummy_comfy_img)

        self.assertEqual(len(pil_imgs), 1)
        self.assertIsInstance(pil_imgs[0], Image.Image)
        self.assertEqual(pil_imgs[0].size, (512, 512))

    def test_pil_to_tensor(self):
        from comfyui_mlx_universal.runtime.bridge import pil_to_tensor
        from PIL import Image
        import torch

        # Create dummy PIL image
        img = Image.new("RGB", (256, 256), color="red")
        
        tensor = pil_to_tensor(img)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (1, 256, 256, 3))
        # Red should be maxed out in the first channel
        self.assertAlmostEqual(tensor[0, 0, 0, 0].item(), 1.0, places=2)
        self.assertAlmostEqual(tensor[0, 0, 0, 1].item(), 0.0, places=2)

if __name__ == "__main__":
    unittest.main()
