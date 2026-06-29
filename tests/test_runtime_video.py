import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Import the helper module first to ensure mocks are in place if needed
import tests.test_helper
from comfyui_mlx_universal.runtime.video_processing import _build_video_cmd, execute_video_generation

class TestRuntimeVideo(unittest.TestCase):
    
    def test_build_video_cmd_wan(self):
        cmd = _build_video_cmd(
            cmd_family="wan",
            model_repo_or_dir="wan-model-repo",
            prompt="A test prompt",
            negative_prompt="bad quality",
            width=512,
            height=512,
            num_frames=16,
            steps=20,
            guide_scale=7.5,
            seed=42,
            output_path="/tmp/output.mp4",
            image=None,
            temp_img_path="/tmp/input.png",
            audio_path=""
        )
        self.assertEqual(cmd[0], sys.executable)
        self.assertEqual(cmd[1:3], ["-m", "mlx_video.wan_2.generate"])
        self.assertIn("--model-dir", cmd)
        self.assertIn("wan-model-repo", cmd)
        self.assertIn("--seed", cmd)
        self.assertIn("42", cmd)
        # Should not have image flag
        self.assertNotIn("--image", cmd)

    def test_build_video_cmd_ltx2_with_audio(self):
        # We need os.path.exists to return True for the audio path check
        with patch('os.path.exists', return_value=True):
            cmd = _build_video_cmd(
                cmd_family="ltx_2",
                model_repo_or_dir="ltx2-repo",
                prompt="A test prompt",
                negative_prompt="",
                width=512,
                height=512,
                num_frames=16,
                steps=20,
                guide_scale=7.5,
                seed=-1,
                output_path="/tmp/output.mp4",
                image=None,
                temp_img_path="/tmp/input.png",
                audio_path="/tmp/test_audio.wav"
            )
            self.assertEqual(cmd[1:3], ["-m", "mlx_video.ltx_2.generate"])
            self.assertIn("--model-repo", cmd)
            self.assertIn("ltx2-repo", cmd)
            self.assertIn("--audio-file", cmd)
            self.assertIn("/tmp/test_audio.wav", cmd)
            self.assertNotIn("--seed", cmd)

    @patch('subprocess.Popen')
    def test_execute_video_generation_failure(self, mock_popen):
        # Setup mock process that fails immediately
        mock_process = MagicMock()
        mock_process.stdout.read.return_value = ""
        mock_process.poll.return_value = 1  # Exit code 1
        mock_popen.return_value = mock_process
        
        with self.assertRaisesRegex(RuntimeError, "process failed with exit code 1"):
            execute_video_generation(
                model_repo_or_dir="some-repo",
                prompt="test",
                negative_prompt="test",
                width=512,
                height=512,
                num_frames=8,
                steps=10,
                guide_scale=3.5,
                seed=42,
                temp_dir="/tmp",
            )

if __name__ == "__main__":
    unittest.main()
