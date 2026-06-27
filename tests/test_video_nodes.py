import unittest
from unittest.mock import MagicMock

from tests.test_helper import import_node_module


class TestVideoNodes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.video_nodes = import_node_module("video_nodes")
        cls.MLXVideoGenerator = cls.video_nodes.MLXVideoGenerator

    def test_generate_video_happy_path(self):
        # Configure the mock for execute_video_generation
        mock_output = ("/tmp/output_video.mp4", "mock_images_tensor")
        self.video_nodes.execute_video_generation = MagicMock(return_value=mock_output)

        # Setup mock node
        node = self.MLXVideoGenerator()

        result = node.generate_video(
            model_repo_or_dir="mlx-community/LTX-2-dev-bf16",
            prompt="A happy dog running",
            negative_prompt="blurry",
            width=512,
            height=512,
            num_frames=16,
            steps=20,
            guide_scale=3.0,
            seed=42,
            image=None,
            audio_path="",
        )

        # Asserts
        self.video_nodes.execute_video_generation.assert_called_once()
        call_kwargs = self.video_nodes.execute_video_generation.call_args[1]

        self.assertEqual(
            call_kwargs["model_repo_or_dir"], "mlx-community/LTX-2-dev-bf16"
        )
        self.assertEqual(call_kwargs["prompt"], "A happy dog running")
        self.assertEqual(call_kwargs["negative_prompt"], "blurry")
        self.assertEqual(call_kwargs["width"], 512)
        self.assertEqual(call_kwargs["height"], 512)
        self.assertEqual(call_kwargs["num_frames"], 16)
        self.assertEqual(call_kwargs["steps"], 20)
        self.assertEqual(call_kwargs["guide_scale"], 3.0)
        self.assertEqual(call_kwargs["seed"], 42)
        self.assertEqual(call_kwargs["image"], None)
        self.assertEqual(call_kwargs["audio_path"], "")

        self.assertEqual(result, mock_output)


if __name__ == "__main__":
    unittest.main()
