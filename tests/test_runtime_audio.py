import os
import unittest
from unittest.mock import MagicMock, patch

import torch
from comfyui_mlx_universal.runtime.audio_processing import (
    execute_audio_transcription,
    execute_tts_generation,
)


@unittest.skipIf(
    os.environ.get("REAL_MLX_TESTS") == "1", "Skipping mock tests with real MLX"
)
class TestRuntimeAudio(unittest.TestCase):
    @patch("mlx_whisper.transcribe")
    def test_execute_audio_transcription(self, mock_transcribe):
        mock_transcribe.return_value = {"text": "Hello world!"}

        audio_dict = {"waveform": torch.zeros((1, 1, 16000)), "sample_rate": 16000}

        result = execute_audio_transcription(
            audio_dict, "mlx-community/whisper-large-v3-turbo"
        )
        self.assertEqual(result, "Hello world!")

        # Verify the mock was called correctly
        self.assertTrue(mock_transcribe.called)

        # Extract the args
        args, kwargs = mock_transcribe.call_args
        self.assertTrue(args[0].endswith(".wav"))
        self.assertEqual(
            kwargs["path_or_hf_repo"], "mlx-community/whisper-large-v3-turbo"
        )

    @patch("comfyui_mlx_universal.runtime.model_loader.load_kokoro_pipeline")
    @patch("comfyui_mlx_universal.runtime.bridge.mlx_to_torch")
    @patch("mlx.core.array")
    @patch("mlx.core.eval")
    @patch("mlx.core.concatenate")
    def test_execute_tts_generation(
        self, mock_concat, mock_eval, mock_array, mock_bridge, mock_load
    ):
        # Setup mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [("a", "b", "mock_audio_chunk")]
        mock_load.return_value = mock_pipeline

        # Setup mock bridge
        mock_tensor = MagicMock()
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.dim.return_value = 1
        mock_bridge.return_value = mock_tensor

        result = execute_tts_generation("hello", "af_heart", 1.0)

        mock_load.assert_called_once_with("prince-canuma/Kokoro-82M")
        mock_pipeline.assert_called_once_with(
            "hello", voice="af_heart", speed=1.0, split_pattern=r"\n+"
        )
        mock_bridge.assert_called_once()

        self.assertIn("waveform", result)
        self.assertEqual(result["sample_rate"], 24000)


if __name__ == "__main__":
    unittest.main()
