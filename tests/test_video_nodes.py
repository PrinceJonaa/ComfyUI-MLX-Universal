import unittest
from unittest.mock import patch

from tests.test_helper import import_node_module


class TestVideoNodes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.video_nodes = import_node_module("video_nodes")
        cls.MLXVideoGenerator = cls.video_nodes.MLXVideoGenerator

    def test_node_types_and_schema(self):
        # 1) Test INPUT_TYPES contains all required keys
        input_types = self.MLXVideoGenerator.INPUT_TYPES()
        required_keys = [
            "model_repo_or_dir",
            "prompt",
            "width",
            "height",
            "num_frames",
            "steps",
            "guide_scale",
            "seed",
        ]
        for key in required_keys:
            self.assertIn(key, input_types["required"])

        # 2) Test RETURN_TYPES equals ('STRING', 'IMAGE')
        self.assertEqual(self.MLXVideoGenerator.RETURN_TYPES, ("STRING", "IMAGE"))

    def test_generate_video_happy_path(self):
        with patch.object(self.video_nodes, "execute_video_generation") as mock_execute:
            # Configure the mock to return fake tuple matching RETURN_TYPES
            mock_output = ("/tmp/fake_video.mp4", None)
            mock_execute.return_value = mock_output

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

            # 3) Test that generate_video() calls execute_video_generation with the correct kwargs
            mock_execute.assert_called_once()
            call_kwargs = mock_execute.call_args[1]

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

            # Verify output matches
            self.assertEqual(result, mock_output)


if __name__ == "__main__":
    unittest.main()
